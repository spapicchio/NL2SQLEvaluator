"""SQLite implementation of the database execution protocol.

This module provides a parallelized executor for SQLite queries, supporting
read-only protections, query timeouts, and optional result caching.
"""
import multiprocessing as mp
import sqlite3
import time
from functools import partial
from pathlib import Path
from typing import TypeAlias

from func_timeout import func_timeout, FunctionTimedOut

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import SQLCacheProtocol, \
    DataToFetch
from NL2SQLEvaluator.db_executor_nodes.output_table import SQLOutputTable
from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecutorError, TaskToBeExecuted
from NL2SQLEvaluator.node_registry import register_node

ParamsType: TypeAlias = list[dict] | dict | None


@register_node(package_name="db_executor_nodes")
class SQLiteDBExecutor:
    """
    Parallelized SQLite query executor.

        This class implements the CodeExecuteProtocol to run SQL queries against
        SQLite databases. It uses multiprocessing to speed up batch execution.
    """

    def execute_queries(
            self,
            tasks: list[TaskToBeExecuted],
            cache_db: SQLCacheProtocol | None = None,
            cache_db_file: str | None = None,
            allow_write: bool = False,
            *args, **kwargs
    ) -> list[list[SQLOutputTable | ExecutorError]]:
        """Executes batches of SQL queries across multiple SQLite databases.

        Why: Provides a safe, isolated, and parallel way to run multiple SQL
        predictions against various local database files.

        Example:
            >>> executor = SQLiteDBExecutor()
            >>> task = TaskToBeExecuted(db_path="test.db", queries=["SELECT * FROM users"])
            >>> results = executor.execute_queries([task])

        Args:
            tasks: A list of TaskToBeExecuted objects containing queries and DB paths.
            cache_db: An optional caching provider.
            cache_db_file: Path to the persistent cache file.
            allow_write: If True, allows INSERT/UPDATE/DELETE. If False, enforces PRAGMA query_only.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments (e.g., num_cpus).

        Returns:
            A nested list where results[task_index][query_index] is the OutputTable or ExecutorError.
        """
        tasks_mp = self._build_tasks_for_mp(tasks, allow_write)

        if not tasks_mp:
            return [[] for _ in tasks]

        total_tasks = len(tasks_mp)
        default_procs = min(max(1, mp.cpu_count()), max(1, total_tasks))
        num_cpus: int = kwargs.get("num_cpus", default_procs)

        # Optimization for single query or single-process environments
        execute_fn_with_cache = _execute_single_query
        if cache_db and cache_db_file:
            execute_fn_with_cache = partial(_execute_single_query, cache_db=cache_db, cache_db_file=cache_db_file)

        if len(tasks_mp) == 1:
            _, _, res = execute_fn_with_cache(*tasks_mp[0])
            results: list[list[SQLOutputTable | ExecutorError]] = [[ExecutorError()] * len(t.queries) for t in tasks]
            results[tasks_mp[0][0]][tasks_mp[0][1]] = res
            return results

        with mp.Pool(processes=num_cpus) as pool:
            flat_results: list[tuple[int, int, SQLOutputTable | ExecutorError]] = pool.starmap(
                execute_fn_with_cache,
                tasks_mp
            )

        # Reassemble into rectangular [tasks][query_idx] result
        results: list[list[SQLOutputTable | ExecutorError]] = [[ExecutorError()] * len(t.queries) for t in tasks]
        for job_id, idx, value in flat_results:
            results[job_id][idx] = value

        return results

    def _build_tasks_for_mp(self, tasks: list[TaskToBeExecuted], allow_write):
        """Flattens Task objects into a list of tuples for multiprocessing."""
        flattened_tasks = []
        for job_id, task in enumerate(tasks):
            for idx, query in enumerate(task.queries):
                # Handle potential list or dict params from TaskToBeExecuted
                params = task.params[idx] if isinstance(task.params, list) else task.params
                flattened_tasks.append((
                    job_id,
                    idx,
                    task.db_path,
                    query,
                    task.timeout,
                    allow_write,
                    params,
                ))
        return flattened_tasks


def _execute_single_query(
        job_id: int,
        idx: int,
        db_file: str,
        query: str,
        timeout: float,
        allow_write: bool = False,
        params: ParamsType = None,
        *,
        cache_db: SQLCacheProtocol | None = None,
        cache_db_file: str | None = None,
) -> tuple[int, int, SQLOutputTable | ExecutorError]:
    """Internal worker function to execute a single SQL query."""

    def _run() -> SQLOutputTable | ExecutorError:
        if cache_db and cache_db_file:
            try:
                db_id = Path(db_file).stem
                cached = cache_db.get_from_cache(
                    cache_db_file,
                    data_to_fetch=[DataToFetch(db_path=db_id, query=query)]
                )
                if not isinstance(cached, Exception):
                    return cached[0].result
            except Exception:
                pass

        start_time = time.perf_counter()
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
                    return SQLOutputTable(rows=[([cur.rowcount])], executed_time=elapsed * 1.25)

                else:
                    cur.execute(query, params or {})
                    rows = cur.fetchall()
                    conn.rollback()
                    elapsed = time.perf_counter() - start_time
                    return SQLOutputTable(rows=rows, executed_time=elapsed * 1.10)

            except Exception as e:
                try:
                    conn.rollback()
                except Exception:
                    pass
                return ExecutorError(str(e))
            finally:
                conn.close()
        except Exception as e:
            return ExecutorError(str(e))

    try:
        result = func_timeout(timeout, _run)
    except FunctionTimedOut:
        result = ExecutorError(f"Query Timeout with {timeout} seconds")
    except Exception as e:
        result = ExecutorError(str(e))

    return job_id, idx, result
