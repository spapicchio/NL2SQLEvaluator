"""
Utilities to augment SQL DDL (CREATE TABLE) strings with example data.

This module provides helpers to:
- Fetch sample rows for a given SQLAlchemy `Table` using a user-provided execute function.
- Inject example values inline as comments next to column definitions.
- Append INSERT statements with sampled rows after the DDL.

The sampling relies on a callback (`execute_fn`) that executes SQL against the
target database, keeping this module independent of any specific DB executor.
"""

import re
from typing import Callable, Literal, Optional

from sqlalchemy import insert, Dialect
from sqlalchemy.sql.schema import Table


def utils_augment_ddl(
        ddl: str,
        table: Table,
        execute_fn: Callable,
        dialect: Dialect,
        strategy: Optional[Literal["append", "inline"]] = None,
        num_rows: int = 1):
    """Augment a DDL string with example data, either inline or appended.

    Depending on `strategy`, this function will:
    - return the original DDL unchanged (`None`),
    - inject example values inline as comments next to column definitions (`"inline"`),
    - append INSERT statements produced from sampled rows (`"append"`).

    Args:
        ddl (str): The CREATE TABLE DDL to augment.
        table (Table): SQLAlchemy table object associated with the DDL.
        execute_fn (Callable): Callable used to execute SQL and fetch sample rows.
            It should accept keyword argument `query` with a SQL string.
        dialect (Dialect): SQLAlchemy dialect for compiling INSERT statements.
        strategy (Optional[Literal["append", "inline"]]): Augmentation strategy.
        num_rows (int): Number of rows to sample for examples.

    Returns:
        str: The augmented DDL string.
    """
    if strategy is None:
        return ddl
    elif strategy == "inline":
        return _utils_augment_ddl_inline_rows(ddl, table, execute_fn, num_rows)
    elif strategy == "append":
        return _utils_augment_ddl_append_rows(ddl, table, execute_fn, dialect, num_rows)

    return ddl


def _utils_select_rows_from(table: Table, execute_fn: Callable, num_rows: int = 1, ):
    """Fetch sample rows from a table using the provided execution function.

    Args:
        table (Table): SQLAlchemy table to sample from.
        execute_fn (Callable): Callable used to execute SQL and fetch rows.
            It should accept keyword argument `query` with a SQL string.
        num_rows (int): Number of rows to retrieve.

    Notes:
        If `execute_fn` raises an exception, this function will catch it and return an empty list.
        This allows callers to handle sampling failures gracefully.

    Returns:
        Any: Result of `execute_fn`, expected to be an iterable of rows.
    """
    try:
        sample_rows_result = execute_fn(
            query=f"SELECT * FROM `{table.name}` LIMIT {num_rows}",
        )
    except Exception as e:
        # Re-raise to let callers decide how to handle failures during sampling.
        return []
    return sample_rows_result


def _utils_augment_ddl_inline_rows(ddl: str, table: Table, execute_fn: Callable, num_rows: int = 1):
    """
    Inject example values as inline comments into a CREATE TABLE statement.

    Args:
        ddl (str): Stringified CREATE TABLE statement.
        table (Table): SQLAlchemy table to sample from.
        execute_fn (Callable): Function to execute SQL queries.
        num_rows (int): Number of rows to inspect for examples.

    Returns:
        str: Modified CREATE TABLE DDL with inline example comments.
    """

    sample_rows = list(_utils_select_rows_from(table, execute_fn, num_rows=num_rows))
    if len(sample_rows) == 0:
        return ddl

    col_examples = {}
    for idx, col in enumerate(table.columns):
        # truncate long values for readability in comments
        examples = {str(row[idx])[:100] for row in sample_rows if row[idx] is not None}
        if examples:
            col_examples[col.name] = f"Example Values: {tuple(examples)}"

    ddl_lines = ddl.splitlines()
    new_lines = []
    for line in ddl_lines:
        # Matches a column definition line to extract the column name.
        # Pattern explanation:
        #   \s*            -> optional leading whitespace
        #   ([`"\[]?)      -> optional opening quote/backtick/bracket
        #   (\w+)          -> column name (captured)
        #   \1             -> optional matching closing quote/backtick/bracket
        #   \s+[\w\(\)]+.* -> at least one space, then the type and the rest of the line
        match = re.match(r"\s*([`\"\[]?)(\w+)\1\s+[\w\(\)]+.*", line)
        if match:
            col_name = match.group(2)
            if col_name in col_examples:
                if line.rstrip().endswith(","):
                    line = line.rstrip()[:-1]
                    line += f" -- {col_examples[col_name]},"
                else:
                    line += f" -- {col_examples[col_name]}"
        new_lines.append(line)
    return "\n".join(new_lines)


def _utils_augment_ddl_append_rows(ddl: str, table: Table, execute_fn: Callable, dialect: Dialect, num_rows: int = 1):
    """
    Render INSERT statements with sampled rows for a table.

    Args:
        ddl (str): Stringified CREATE TABLE statement.
        table (Table): SQLAlchemy table to sample from.
        execute_fn (Callable): Function to execute SQL queries.
        dialect (Dialect): SQLAlchemy dialect for compiling statements.
        num_rows (int): Number of rows to sample and render as INSERTs.

    Returns:
        str: One INSERT statement per sampled row.
    """
    sample_rows = list(_utils_select_rows_from(table, execute_fn, num_rows=num_rows))

    inserts = []
    for row in sample_rows:
        # Truncate long values for readability in generated INSERTs.
        row = [str(i)[:100] for i in row]
        stmt = insert(table).values(dict(zip(table.columns.keys(), row)))
        compiled = stmt.compile(
            dialect=dialect, compile_kwargs={"literal_binds": True}
        )
        inserts.append(str(compiled))

    inserts = "\n".join(inserts)
    return f"{ddl}\n{inserts}" if inserts else ddl
