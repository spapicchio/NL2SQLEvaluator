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

import re
from collections import defaultdict
from typing import Protocol, Any, Self, List, Optional, runtime_checkable

from pydantic import BaseModel, model_validator, ConfigDict


class ExecutorError(Exception):
    """Raised when a database execution fails or times out."""
    pass


class TaskToBeExecuted(BaseModel):
    """Represents a set of queries to be executed on a specific database."""
    model_config = ConfigDict(extra='allow')
    db_path: str
    queries: list[str]
    db_id: Optional[str] = None
    params: Optional[list[dict | tuple] | dict | tuple] = None
    timeout: float = 500.0

    @model_validator(mode='after')
    def broadcast_params(self) -> Self:
        """Ensures params match the number of queries."""
        if self.params is None or isinstance(self.params, (dict, tuple)) or self.params == []:
            p_val = self.params or {}
            self.params = [p_val for _ in range(len(self.queries))]

        if len(self.params) != len(self.queries):
            raise ValueError("Length of params must match length of queries.")
        return self


@runtime_checkable
class CodeExecuteProtocol(Protocol):
    """Interface for database execution engines."""

    def execute_queries(
            self,
            tasks: List[TaskToBeExecuted],
            cache_db: Optional[Any] = None,
            cache_db_file: Optional[str] = None,
            *args: Any,
            **kwargs: Any
    ) -> list[list[Any | ExecutorError]]:
        ...


# TODO move from this module
def extract_last_match(text: str, pattern: str) -> str:
    """Extracts the last match of a regex pattern or returns the original string."""
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else text


# TODO move from this module
def utils_extract_sql_or_same(generation: str) -> str:
    """Parses SQL from LLM responses, looking for <answer> or code blocks."""
    content = extract_last_match(generation, r"<answer>(.*?)</answer>")
    content = extract_last_match(content, r"```sql\s*(.*?)\s*```")
    content = extract_last_match(content, r"```\s*(.*?)\s*```")
    return content.strip().strip("`").strip()


def execute_queries_in_model_predictions(
        code_executor: CodeExecuteProtocol,
        db_files: List[str],
        queries: List[List[str]],
        params: Optional[list[list[dict | tuple] | dict | tuple]] = None,
        cached_db: Optional[Any] = None,
        cache_db_file: Optional[str] = None,
        **kwargs: Any
) -> list[list[Any | ExecutorError]]:
    """Groups predictions by database and dispatches them for execution.

    Args:
        cached_db:
        cache_db_file:
        code_executor: Engine implementing the execution logic.
        db_files: List of DB paths (determines the group).
        queries: Nested list of SQL strings.
        **kwargs: Configuration for TaskToBeExecuted (e.g., timeout).

    Returns:
        List of results matched to the input order of db_files.
    """
    if not isinstance(code_executor, CodeExecuteProtocol):
        raise TypeError(f"Object {type(code_executor).__name__} does not implement CodeExecuteProtocol")

    # storage maps db_path -> { queries_list, bookkeeping_info }
    storage = defaultdict(lambda: {"queries": [], "params": [], "bookkeeping": []})

    for i, db_path in enumerate(db_files):
        group = storage[db_path]
        queries_to_execute = queries[i]

        start = len(group["queries"])
        end = start + len(queries_to_execute)

        group["queries"].extend(queries_to_execute)
        group["params"].extend(params[i] if params else ())

        # original_position: Where this batch sits in the user's input list
        # result_segment: Which part of the combined results belongs to this batch
        group["bookkeeping"].append({
            "original_position": i,
            "result_segment": slice(start, end)
        })

    # Execute all unique database tasks
    storage_items = list(storage.items())
    tasks = [
        TaskToBeExecuted(db_path=p, queries=v["queries"], params=v["params"], **kwargs)
        for p, v in storage_items
    ]
    all_results = code_executor.execute_queries(
        tasks=tasks,
        cache_db=cached_db,
        cache_db_file=cache_db_file
    )

    # Reassemble results into the original input order
    final_output = [None] * len(db_files)
    for task_idx, (db_path, data) in enumerate(storage_items):
        # batch_results is the long list of results for one specific database
        batch_results = all_results[task_idx]

        for entry in data["bookkeeping"]:
            pos = entry["original_position"]
            seg = entry["result_segment"]
            final_output[pos] = batch_results[seg]

    return final_output
