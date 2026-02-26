"""
SQLite implementation of the database execution protocol.

Refactored for clarity: Separates cache logic, process management,
and SQL execution into distinct, manageable units.
"""
import multiprocessing as mp
import sqlite3
import time
from pathlib import Path
from typing import List, Optional, Any

from func_timeout import func_timeout, FunctionTimedOut

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import (
    SQLCacheProtocol, DataToFetch, NotFoundInCacheError
)
from NL2SQLEvaluator.db_executor_nodes.db_executor_input import TaskToBeExecuted
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SQLExecutorOutput, ExecutorError
from NL2SQLEvaluator.node_registry import register_node


@register_node(package_name="db_executor_nodes")
class SQLiteDBExecutor:
    """Parallelized SQLite query executor with integrated caching and safety limits."""

    def execute_queries(
            self,
            tasks: List[TaskToBeExecuted],
            allow_write: bool = False,
            cache_db: Optional[SQLCacheProtocol[SQLExecutorOutput]] = None,
            **kwargs
    ) -> list[list[SQLExecutorOutput | ExecutorError]]:
        """Executes SQL tasks. Checks cache first, then runs misses in parallel."""
        results: list[list[SQLExecutorOutput | ExecutorError]] = [
            [ExecutorError("Pending execution")] * len(t.queries) for t in tasks
        ]
        pending_work = []

        # Step 1: Cache Check (Serial is fast here)
        for j_id, task in enumerate(tasks):
            for q_idx, query in enumerate(task.queries):
                # Handle param indexing
                params = task.params[q_idx] if isinstance(task.params, list) else task.params
                if cache_db:
                    cached = self._check_cache(cache_db, task.db_path, query)
                    if cached is not None:
                        results[j_id][q_idx] = cached
                        continue

                timeout = task.timeout[q_idx] if isinstance(task.timeout, list) else task.timeout
                pending_work.append((j_id, q_idx, task.db_path, query, timeout, allow_write, params))

        if len(pending_work) == 0:
            return results

        # Step 2: Parallel Execution
        num_cpus = kwargs.get("num_cpus", min(len(pending_work), mp.cpu_count() or 1))
        with mp.Pool(processes=num_cpus) as pool:
            flat_results = pool.starmap(_execute_single_query, pending_work)

        # Step 3: Reassemble
        for j_id, q_idx, outcome in flat_results:
            results[j_id][q_idx] = outcome

        return results

    def _check_cache(self,
                     cache_db: SQLCacheProtocol[SQLExecutorOutput],
                     db_path: str,
                     query: str) -> SQLExecutorOutput | None:
        """Internal helper to safely probe the cache."""
        db_id = Path(db_path).stem
        fetch_req = DataToFetch(db_path=db_id, query=query)
        cache_res = cache_db.get_from_cache([fetch_req])[0]
        if not isinstance(cache_res, NotFoundInCacheError):
            return cache_res.result
        return None


def _execute_single_query(
        job_id: int,
        idx: int,
        db_file: str,
        query: str,
        timeout: float,
        allow_write: bool,
        params: Any
) -> tuple[int, int, SQLExecutorOutput | ExecutorError]:
    """Isolated worker function to execute a single SQLite query."""

    def _run_sql(db_file_, query_, params_) -> SQLExecutorOutput:
        start = time.perf_counter()
        with sqlite3.connect(db_file_) as conn:
            conn.execute("PRAGMA foreign_keys=ON;")
            if not allow_write:
                conn.execute("PRAGMA query_only=ON;")

            cur = conn.cursor()
            cur.execute(query_, params_ or {})
            affected = cur.rowcount if cur.rowcount != -1 else 0
            rows = cur.fetchall() if not allow_write else [(affected,)]

            # Mocking a slight overhead as in original code
            multiplier = 1.25 if allow_write else 1.10
            elapsed = (time.perf_counter() - start) * multiplier

            return SQLExecutorOutput(
                rows=rows,
                execution_time=elapsed
            )

    try:
        result: SQLExecutorOutput | Any = func_timeout(timeout, _run_sql, args=(db_file, query, params))

    except FunctionTimedOut:
        result = ExecutorError(f"Timeout after {timeout}s")

    except sqlite3.Error as e:
        result = ExecutorError(str(e))

    return job_id, idx, result
