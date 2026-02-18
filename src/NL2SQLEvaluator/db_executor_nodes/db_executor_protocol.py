"""Module for optimized code execution via database grouping.

This module handles the logic of taking multiple code prediction batches,
grouping them by database to minimize connection overhead, and reassembling 
them into the original request order.

Example:
    If input is:
        Batch 0 (DB_A): ["Q1"]
        Batch 1 (DB_B): ["Q2", "Q3"]
        Batch 2 (DB_A): ["Q4"]

    The logic groups them as:
        Task 1 (DB_A): ["Q1", "Q4"] -> Results: [R1, R4]
        Task 2 (DB_B): ["Q2", "Q3"] -> Results: [R2, R3]

    Using 'bookkeeping', it reassembles them to:
        [[R1], [R2, R3], [R4]]
"""

from collections import defaultdict
from typing import Protocol, Any, runtime_checkable
from typing import TypeVar

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import SQLCacheProtocol
from NL2SQLEvaluator.db_executor_nodes.db_executor_input import TaskToBeExecuted
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import ExecutorError

T = TypeVar("T")


@runtime_checkable
class DBExecutorProtocol(Protocol[T]):
    """Interface for database execution engines."""

    def execute_queries(
            self,
            tasks: list[TaskToBeExecuted],
            cache_db: SQLCacheProtocol[T] | None = None,
            cache_db_file: str | None = None,
            *args: Any,
            **kwargs: Any
    ) -> list[list[T | ExecutorError]]:
        ...


def execute_queries_in_model_predictions(
        code_executor: DBExecutorProtocol[T],
        db_paths: list[str],
        queries: list[list[str]],
        params: list[list[dict]] | None = None,
        cached_db: SQLCacheProtocol[T] | None = None,
        cache_db_file: str | None = None,
        **kwargs: Any
) -> list[list[T | ExecutorError]]:
    """Groups SQL predictions by database path to optimize execution.

        Why:
            Executing queries one by one or in small batches across different
            databases is slow due to connection overhead. This function groups
            all queries belonging to the same DB, executes them in one go,
            and maps the results back to the original request order.

        How:
            Provide a list of database paths and a matching list of query batches.
            The function handles the internal bookkeeping and returns results
            aligned with your input list.

        Example:
            >>> results = execute_queries_in_model_predictions(
            ...     executor,
            ...     db_paths=["db1", "db2", "db1"],
            ...     queries=[["SELECT 1"], ["SELECT 2"], ["SELECT 3"]]
            ... )
            >>> # results will be [[res1], [res2], [res3]]

        Args:
            code_executor: The engine used to run the queries.
            db_paths: A list of paths to the databases for each batch.
            queries: A nested list where each inner list contains SQL queries.
            params: Optional nested list of parameter dictionaries for queries.
            cached_db: An optional cache provider.
            cache_db_file: Path to a cache file.
            **kwargs: Additional settings passed to TaskToBeExecuted (e.g., timeout).

        Returns:
            A list of result lists, where the outer index matches the input index.

        Raises:
            TypeError: If code_executor does not conform to CodeExecuteProtocol.
        """
    if not isinstance(code_executor, DBExecutorProtocol):
        raise TypeError(f"Object {type(code_executor).__name__} does not implement CodeExecuteProtocol")

    # Storage maps db_path -> aggregated data
    db_path2queries_to_execute: dict[str, dict[str, list]] = defaultdict(
        lambda: {"queries": [], "params": [], "bookkeeping": []}
    )

    for i, db_path in enumerate(db_paths):
        group = db_path2queries_to_execute[db_path]
        batch_queries = queries[i]

        start = len(group["queries"])
        end = start + len(batch_queries)

        group["queries"].extend(batch_queries)
        # Handle optional params
        batch_params = params[i] if params else [{}] * len(batch_queries)
        group["params"].extend(batch_params)
        group["bookkeeping"].append({
            "original_position": i,
            "result_segment": slice(start, end)
        })

    # Prepare and execute grouped tasks
    storage_items = list(db_path2queries_to_execute.items())
    tasks = [
        TaskToBeExecuted(
            db_path=db_path,
            queries=queries_to_execute["queries"],
            params=queries_to_execute["params"],
            **kwargs
        )
        for db_path, queries_to_execute in storage_items
    ]

    all_results = code_executor.execute_queries(
        tasks=tasks,
        cache_db=cached_db,
        cache_db_file=cache_db_file
    )

    # Initialize final_output with the correct size using the input count
    final_output: list[list[T | ExecutorError]] = [[] for _ in range(len(db_paths))]

    # Map grouped results back to original indices
    for task_idx, (_, data) in enumerate(storage_items):
        batch_results = all_results[task_idx]
        for entry in data["bookkeeping"]:
            pos = entry["original_position"]
            seg = entry["result_segment"]
            final_output[pos] = batch_results[seg]

    return final_output
