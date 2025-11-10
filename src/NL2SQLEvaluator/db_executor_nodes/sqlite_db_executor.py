import multiprocessing as mp
import sqlite3
import time
from pathlib import Path
from typing import Any, TypeAlias

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import NotFoundInCacheError, OutputTable, SQLCacheProtocol, \
    DataToFetch
from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecutorError, ExecuteTask
from NL2SQLEvaluator.node_registry import register_node
from func_timeout import func_timeout, FunctionTimedOut

ParamsType: TypeAlias = list[dict] | dict | None


@register_node()
class SQLiteDBExecutor:
    @staticmethod
    def execute_queries(
            tasks: list[ExecuteTask],
            cache_db: SQLCacheProtocol | None = None,
            cache_db_file: str | None = None,
            allow_write: bool = False,
            *args, **kwargs
    ) -> list[list[OutputTable | ExecutorError]]:

        tasks_mp = _build_task_for_mp(tasks, cache_db, cache_db_file, allow_write)
        total_tasks = len(tasks_mp)
        default_procs = min(max(1, mp.cpu_count()), max(1, total_tasks))
        num_cpus: int = kwargs.get("num_cpus", default_procs)
        if len(tasks_mp) == 1:
            return [[_execute_single_query(*tasks_mp[0])[2]]]
        # Run workers
        with mp.Pool(processes=num_cpus) as pool:
            flat_results: list[tuple[int, int, OutputTable | ExecutorError]] = pool.starmap(
                _execute_single_query,
                tasks_mp
            )
        # Reassemble into rectangular [jobs][idx] result
        results: list[list[OutputTable | ExecutorError]] = [[ExecutorError()] * len(task.queries) for task in tasks]
        for job_id, idx, value in flat_results:
            results[job_id][idx] = value

        return results


def _build_task_for_mp(tasks: list[ExecuteTask], cache_db, cache_db_file, allow_write):
    flattened_tasks = []
    for job_id, task in enumerate(tasks):
        for idx, (query, params, db_file, timeout) in enumerate(task):
            flattened_tasks.append(
                (job_id, idx, db_file, query, timeout, allow_write, params, cache_db, cache_db_file))
    return flattened_tasks


def _execute_single_query(
        job_id: int,
        idx: int,
        db_file: str,
        query: str,
        timeout: float,
        allow_write: bool = False,
        params: ParamsType = None,
        cache_db: SQLCacheProtocol | None = None,
        cache_db_file: str | None = None,
) -> tuple[int, int, OutputTable | ExecutorError]:
    """
    Execute a single SQL statement with a timeout, in an isolated connection.

    Behavior:
      - If a cache is provided and returns a hit, return the cached value.
      - If allow_write is False: PRAGMA query_only=ON, run the query, fetchall, then ROLLBACK.
      - If allow_write is True: run the query; if params is a list with len>1, use executemany;
        COMMIT on success and return cursor.rowcount.
      - On any exception: attempt ROLLBACK and return ExecutorError(e).
    """

    def _run() -> OutputTable | ExecutorError:
        # 1) Cache short-circuit
        cached = _maybe_get_cached(cache_db, cache_db_file, db_file, query)
        if cached is not None:
            return cached
        start_time = time.perf_counter()
        # 2) DB execution
        try:
            conn = sqlite3.connect(db_file)
            try:
                conn.execute("PRAGMA foreign_keys=ON;")
                if not allow_write:
                    conn.execute("PRAGMA query_only=ON;")

                conn.execute("BEGIN TRANSACTION;")
                cur = conn.cursor()

                if allow_write:
                    cur.execute(query, params or {})
                    conn.commit()
                    elapsed = time.perf_counter() - start_time
                    safe_time = elapsed + (elapsed * 0.25)
                    return OutputTable(rows=[([cur.rowcount])], executed_time=safe_time)

                else:
                    # Reads: always single execute + fetchall + rollback
                    cur.execute(query, params or {})
                    rows = cur.fetchall()
                    conn.rollback()
                    elapsed = time.perf_counter() - start_time
                    safe_time = elapsed + (elapsed * 0.10)
                    return OutputTable(rows=rows, executed_time=safe_time)

            except Exception as e:
                # Try rollback, ignore rollback failures
                try:
                    conn.rollback()
                except Exception:
                    pass
                return ExecutorError(e)

            finally:
                conn.close()

        except Exception as e:
            # Connection-level failures
            return ExecutorError(e)

    try:
        result = func_timeout(timeout, _run)
    except FunctionTimedOut:
        result = ExecutorError(f"Query Timeout with {timeout} seconds")

    return job_id, idx, result


# -------------------------
# Small, focused helpers
# -------------------------

def _db_id_from_path(db_file: str) -> str:
    """Derive a stable DB id from a file path (matches previous basename-without-ext logic)."""
    return Path(db_file).stem


def _maybe_get_cached(
        cache_db: SQLCacheProtocol | None,
        cache_db_file: str | None,
        db_file: str,
        query: str,
) -> Any | None:
    """Return cached value if available; None on miss or when cache is not configured."""
    if cache_db is None or cache_db_file is None:
        return None
    db_id = _db_id_from_path(db_file)
    cached = cache_db.get_from_cache(cache_db_file, data_to_fetch=[DataToFetch(db_id=db_id, query=query)])[0]
    if cached and not isinstance(cached, NotFoundInCacheError):
        return cached
    return None
