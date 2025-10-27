import multiprocessing as mp
import sqlite3
from pathlib import Path
from typing import Any, Union

from func_timeout import func_timeout, FunctionTimedOut

from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecutorError, OutputTable, NotFoundInCacheError, \
    SQLCacheProtocol
from NL2SQLEvaluator.node_registry import register_node

ParamsType = Union[list[dict], dict, None]


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

        if len(params) != len(queries):
            raise ValueError("Length of params must match length of queries (or be None).")

        tasks = _build_task_for_mp(db_files, queries, timeout_s, params, cache_db, cache_db_file)
        total_tasks = len(tasks)
        default_procs = min(max(1, mp.cpu_count()), max(1, total_tasks))
        num_cpus: int = kwargs.get("num_cpus", default_procs)
        if len(tasks) == 1:
            return [[_execute_single_query(*tasks[0])[2]]]
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

        # 2) DB execution
        try:
            conn = sqlite3.connect(db_file)
            try:
                _apply_pragmas(conn, allow_write)
                conn.execute("BEGIN TRANSACTION;")
                cur = conn.cursor()

                if allow_write:
                    # Writes: executemany only when params is a list with >1 entries.
                    if _use_executemany(allow_write, params):
                        cur.executemany(query, params)  # type: ignore[arg-type]
                    else:
                        cur.execute(query, _normalized_params(params))
                    conn.commit()
                    return [([cur.rowcount])]
                else:
                    # Reads: always single execute + fetchall + rollback
                    cur.execute(query, _normalized_params(params))
                    rows = cur.fetchall()
                    conn.rollback()
                    return rows

            except Exception as e:
                # Try rollback, ignore rollback failures
                _safe_rollback(conn)
                return ExecutorError(e)
            finally:
                conn.close()

        except Exception as e:
            # Connection-level failures
            return ExecutorError(e)

    try:
        result = func_timeout(timeout_s, _run)
    except FunctionTimedOut:
        result = ExecutorError(f"Query Timeout with {timeout_s} seconds")

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
    cached = cache_db.get_from_cache(cache_db_file, [db_id], [query])[0]
    if cached and not isinstance(cached, NotFoundInCacheError):
        return cached
    return None


def _apply_pragmas(conn: sqlite3.Connection, allow_write: bool) -> None:
    """Apply connection-level PRAGMAs."""
    conn.execute("PRAGMA foreign_keys=ON;")
    if not allow_write:
        conn.execute("PRAGMA query_only=ON;")


def _use_executemany(allow_write: bool, params: ParamsType) -> bool:
    """True when we should call executemany (only for writes with multiple param sets)."""
    return bool(
        allow_write
        and isinstance(params, list)
        and len(params) > 1
    )


def _normalized_params(params: ParamsType) -> dict:
    """Map None -> {}, pass dict as-is; if a list is supplied here, the caller should use executemany."""
    if params is None:
        return {}
    if isinstance(params, dict):
        return params
    # If a list mistakenly reaches here, behave like previous code path would (use empty dict).
    return {}


def _safe_rollback(conn: sqlite3.Connection) -> None:
    try:
        conn.rollback()
    except Exception:
        pass
