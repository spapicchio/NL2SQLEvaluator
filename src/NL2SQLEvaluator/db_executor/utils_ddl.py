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
from collections import defaultdict
from typing import Callable, Literal, Optional

from sqlalchemy import insert, Dialect
from sqlalchemy.sql.expression import select
from sqlalchemy.sql.schema import Table


def utils_augment_ddl_tbl(
        ddl: str,
        table: Table,
        execute_fn: Callable,
        dialect: Dialect,
        strategy: Optional[Literal["append", "inline"]] = None,
        col2values_sim_quest: Optional[dict[str, list[str]]] = None,
        augment_fk_constraints: bool = True,
        col2descr: Optional[dict] = None,
        num_rows: int = 3):
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
        col2values_sim_quest (Optional[dict[str, list[str]]]): Optional mapping of column names to sample values
            provided by an external source. If provided, these values will be used for sampling instead of querying the database.
        col2descr (Optional[dict]): Optional mapping of column names to descriptions.
        augment_fk_constraints (bool): Whether to augment foreign key constraints with standardized names.

    Returns:
        str: The augmented DDL string.
    """
    if strategy is None:
        pass
    elif strategy == "inline":
        ddl = _utils_augment_ddl_inline_rows(ddl, table, execute_fn, num_rows, col2values_sim_quest, col2descr)
    elif strategy == "append":
        ddl = _utils_augment_ddl_append_rows(ddl, table, execute_fn, dialect, num_rows, col2values_sim_quest)

    if augment_fk_constraints:
        ddl = _utils_augment_fk_table_name(ddl, table)

    return ddl


def _utils_select_not_null_samples(table: Table, execute_fn: Callable, num_rows: int = 3,
                                   col2values_sim_quest: Optional[dict[str, list[str]]] = None):
    """Fetch sample rows from a table using the provided execution function.

    Args:
        table (Table): SQLAlchemy table to sample from.
        execute_fn (Callable): Callable used to execute SQL and fetch rows.
            It should accept keyword argument `query` with a SQL string.
        num_rows (int): Number of rows to retrieve.
        col2values_sim_quest (Optional[dict[str, list[str]]]): Optional mapping of column names to sample values
            provided by an external source. If provided, these values will be used for sampling instead of querying the database.

    Notes:
        If `execute_fn` raises an exception, this function will catch it and return an empty list.
        This allows callers to handle sampling failures gracefully.

    Returns:
        Any: Result of `execute_fn`, expected to be an iterable of rows.
    """
    try:
        col2samples = defaultdict(list)
        for col in table.columns:
            if col2values_sim_quest and col.name in col2values_sim_quest:
                # use provided sample values from sim_quest
                col2samples[col.name] = col2values_sim_quest[col.name][:num_rows]
                continue

            # all non-NULL values (duplicates kept)
            stmt = select(col).distinct().where(col.is_not(None)).limit(num_rows)
            col2samples[col.name] = [val[0] for val in execute_fn(stmt)]

        # apply transpose to get rows
        sample_rows = []
        for i in range(num_rows):
            row = []
            for col in table.columns:
                possible_values = col2samples[col.name]
                row.append(possible_values[i % len(possible_values)]) if len(possible_values) > 0 else row.append(None)
            sample_rows.append(tuple(row))
        return sample_rows
    except Exception as e:
        # Re-raise to let callers decide how to handle failures during sampling.
        return []


def _utils_augment_ddl_inline_rows(ddl: str, table: Table, execute_fn: Callable, num_rows: int = 1,
                                   col2values_sim_quest: Optional[dict[str, list[str]]] = None,
                                   col2descr: Optional[dict] = None) -> str:
    """
    Inject example values as inline comments into a CREATE TABLE statement.

    Args:
        ddl (str): Stringified CREATE TABLE statement.
        table (Table): SQLAlchemy table to sample from.
        execute_fn (Callable): Function to execute SQL queries.
        num_rows (int): Number of rows to inspect for examples.
        col2values_sim_quest (Optional[dict[str, list[str]]]): Optional mapping of column names to sample values
            provided by an external source. If provided, these values will be used for sampling instead of
            querying the database.
        col2descr (Optional[dict]): Optional mapping of column names to descriptions.

    Returns:
        str: Modified CREATE TABLE DDL with inline example comments.
    """
    column_def_re = re.compile(
        r"""
        ^\s*
        (?!PRIMARY\b|FOREIGN\b|UNIQUE\b|CONSTRAINT\b|CHECK\bCREATE\b)   # skip table constraints
        (?:
            " (?P<dq>(?:[^"]|"")+ ) "        |   # "quoted", allow "" inside
            \[ (?P<br>[^\]]+) \]             |   # [bracketed]
            ` (?P<bq>[^`]+) `                |   # `backticked`
            (?P<plain>[A-Za-z_][A-Za-z0-9_\$]*)  # unquoted
        )
        \s+                                   # space before type
        [A-Za-z]                              # rudimentary type token start
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    sample_rows = list(
        _utils_select_not_null_samples(table, execute_fn, num_rows=num_rows, col2values_sim_quest=col2values_sim_quest)
    )
    if len(sample_rows) == 0:
        return ddl

    col_examples = {}
    for idx, col in enumerate(table.columns):
        # truncate long values for readability in comments
        examples = {str(row[idx])[:50] for row in sample_rows if row[idx] is not None}
        if examples:
            if col.name in col2descr and col2descr[col.name]:
                col_examples[col.name] = f"{col2descr[col.name]}, example: {list(examples)}"
            else:
                col_examples[col.name] = f"example: {list(examples)}"

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
        m = column_def_re.match(line)
        if not m:
            new_lines.append(line)
            continue

        # Pick whichever group matched
        col_name = next(g for g in (m.group("dq"), m.group("br"), m.group("bq"), m.group("plain")) if g)
        # Unescape doubled quotes if any (SQL standard)
        if m.lastgroup == "dq":
            col_name = col_name.replace('""', '"')

        if col_name in col_examples:
            if line.rstrip().endswith(","):
                line = line.rstrip()[:-1] + f" -- {col_examples[col_name]},"
            else:
                line = line.rstrip() + f" -- {col_examples[col_name]}"
        new_lines.append(line)

    return "\n".join(new_lines)


def _utils_augment_ddl_append_rows(ddl: str, table: Table, execute_fn: Callable, dialect: Dialect, num_rows: int = 1,
                                   col2values_sim_quest: Optional[dict[str, list[str]]] = None):
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
    sample_rows = list(
        _utils_select_not_null_samples(table, execute_fn, num_rows=num_rows, col2values_sim_quest=col2values_sim_quest))

    inserts = []
    for row in sample_rows:
        # Truncate long values for readability in generated INSERTs.
        row = [str(i)[:50] for i in row]
        stmt = insert(table).values(dict(zip(table.columns.keys(), row)))
        compiled = stmt.compile(
            dialect=dialect, compile_kwargs={"literal_binds": True}
        )
        inserts.append(str(compiled))

    inserts = "\n".join(inserts)
    return f"{ddl}\n{inserts}" if inserts else ddl


def _utils_augment_fk_table_name(ddl: str, table: Table) -> str:
    def _internal(line_ddl_fk):
        src = table.name
        m = re.search(
            r'FOREIGN\s+KEY\s*\([^)]+\)\s*REFERENCES\s+([`"\[]?)([^`"\[\]\s(]+(?:\.[^`"\[\]\s(]+)?)\1',
            line_ddl_fk,
            flags=re.IGNORECASE
        )
        if not m:
            return line_ddl_fk
        # Extract and normalize the referenced table name
        dest_raw = m.group(2)  # may be schema-qualified like main.schools
        dest_simple = dest_raw.split('.')[-1].strip().strip('`"[]').lower().replace(" ", "_")
        src_norm = src.lower().replace(" ", "_")
        line_ddl_fk = f"\tCONSTRAINT fk_{src_norm}_{dest_simple} {line_ddl_fk.strip()}"
        return line_ddl_fk

    return "\n".join([_internal(line) for line in ddl.splitlines()])
