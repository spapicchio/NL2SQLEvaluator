import multiprocessing as mp
import os
import sqlite3

from func_timeout import func_timeout, FunctionTimedOut

from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecutorError, OutputTable, NotFoundInCacheError, \
    SQLCacheProtocol
from NL2SQLEvaluator.node_registry import register_node


@register_node()
class SQLiteDBExecutor:
    @staticmethod
    def execute_queries(
            db_files: list[str],
            queries: list[list[str]],
            params: list[dict] | None = None,
            cache_db: SQLCacheProtocol | None = None,
            cache_db_file: str | None = None,
            *args, **kwargs
    ) -> list[list[OutputTable | ExecutorError]]:
        timeout_s = kwargs.get("timeout", 10)

        if not queries:
            return [[]]

        # Normalize per-job params length & default to None
        if params is None:
            params = [None] * len(queries)
        elif len(params) != len(queries):
            raise ValueError("Length of params must match length of queries (or be None).")

        tasks = _build_task_for_mp(db_files, queries, timeout_s, params, cache_db, cache_db_file)
        total_tasks = len(tasks)
        default_procs = min(max(1, mp.cpu_count()), max(1, total_tasks))
        num_cpus: int = kwargs.get("num_cpus", default_procs)

        # Run workers
        with mp.Pool(processes=num_cpus) as pool:
            flat_results: list[tuple[int, int, OutputTable | ExecutorError]] = pool.starmap(
                _execute_single_query,
                tasks
            )
        # Reassemble into rectangular [jobs][idx] result
        results: list[list[OutputTable | ExecutorError]] = [[None] * len(job) for job in queries]  # type: ignore
        for job_id, idx, value in flat_results:
            results[job_id][idx] = value

        return results


def _build_task_for_mp(db_files: list[str], queries: list[list], timeout_s, params, cache_db, cache_db_file):
    tasks = []
    for job_id, (query, param, db_file) in enumerate(zip(queries, params, db_files)):
        for idx, query_i in enumerate(query):
            tasks.append((job_id, idx, db_file, query_i, timeout_s, False, param, cache_db, cache_db_file))
    return tasks


def _execute_single_query(
        job_id: int,
        idx: int,
        db_file: str,
        query: str,
        timeout_s: float,
        allow_write: bool = False,
        params: list[dict] | dict | None = None,
        cache_db: SQLCacheProtocol | None = None,
        cache_db_file: str | None = None,
) -> tuple[int, int, OutputTable | ExecutorError]:
    """
    Executes a single statement with timeout, in its own connection.
    - If allow_write is False, pragma query_only + rollback after SELECT.
    - If allow_write is True, commit on success and return rowcount.
    """

    def _internal():
        db_id = os.path.splitext(os.path.basename(db_file))[0]
        cached = cache_db.get_from_cache(cache_db_file, [db_id], [query])[0]
        if not isinstance(cached, NotFoundInCacheError):
            result = cached
        else:
            conn = None
            try:
                conn = sqlite3.connect(db_file)
                # Always enable foreign keys
                conn.execute("PRAGMA foreign_keys=ON;")
                # If writes are not allowed, try to enable query_only (may not be supported)
                if not allow_write:
                    conn.execute("PRAGMA query_only=ON;")
                cur = conn.cursor()
                conn.execute("BEGIN TRANSACTION;")

                if params is not None and isinstance(params, list):
                    conn.executemany(query, params)
                else:
                    cur.execute(query, params)

                if allow_write:
                    # commit modifications
                    conn.commit()
                    result = cur.rowcount
                else:
                    # fetch read results and then rollback to avoid persisting any changes
                    result = cur.fetchall()
                    conn.rollback()
            except Exception as e:
                # ensure rollback on any error
                try:
                    if conn is not None:
                        conn.rollback()
                except Exception:
                    pass
                result = ExecutorError(e)

            finally:
                if conn is not None:
                    conn.close()
        return result

    try:
        rows = func_timeout(timeout_s, _internal)
    except FunctionTimedOut:
        rows = ExecutorError(f'Query Timeout with {timeout_s} seconds')
    return job_id, idx, rows
