"""
Base interfaces and helpers for SQL DB executors.

This module defines the abstract base class used to implement concrete SQL database
executors and a task function for batch execution.

External API:
- db_executor_worker
- BaseSQLDBExecutor (class)
  - BaseSQLDBExecutor.from_uri
  - BaseSQLDBExecutor.execute_query
  - BaseSQLDBExecutor.execute_multiple_query
  - BaseSQLDBExecutor.execute_query_with_cache
  - BaseSQLDBExecutor.get_table_info
  - BaseSQLDBExecutor.table_names
  - BaseSQLDBExecutor.dialect
  - BaseSQLDBExecutor.db_id
  - BaseSQLDBExecutor.inspector
"""
import atexit
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Literal, Any

import bm25s
import pandas as pd
import sqlalchemy as sa
from langgraph.func import task
from sqlalchemy import Engine, inspect, MetaData
from sqlalchemy import sql
from sqlalchemy.sql.ddl import CreateTable
from sqlalchemy.sql.sqltypes import NullType, String, Text

from NL2SQLEvaluator.db_executor.utils_create_bm25_index import create_bm25_index, retrieve_from_bm25_index
from NL2SQLEvaluator.db_executor.utils_ddl import utils_augment_ddl_tbl
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.orchestrator_state import SQLInstance, SingleTask


@task()
def db_executor_worker(single_task: SingleTask) -> SingleTask:
    """
    Execute target and predicted SQL queries for a task.

    Uses the task's engine to execute queries and leverages caching if configured.

    Args:
        single_task (SingleTask): Task containing `target_sql`, `predicted_sql`, and an `engine`.

    Returns:
        SingleTask: A copy of the task with `target_sql.executed` and `predicted_sql.executed` populated.
    """
    engine = single_task.engine

    executed_target = engine.execute_query_with_cache(single_task.target_sql.query) \
        if single_task.target_sql.executed is None else single_task.target_sql.executed

    executed_predicted = engine.execute_query_with_cache(single_task.predicted_sql.query) \
        if single_task.predicted_sql.executed is None else single_task.predicted_sql.executed

    target = SQLInstance(query=single_task.target_sql.query, executed=executed_target)
    predicted = SQLInstance(query=single_task.predicted_sql.query, executed=executed_predicted)

    return SingleTask(
        target_sql=target,
        predicted_sql=predicted,
        **single_task.model_dump(exclude={"target_sql", "predicted_sql"})
    )


class BaseSQLDBExecutor(ABC):
    """
    Abstract base class for SQL database executors.

    Responsibilities:
        * Provide a unified interface to execute single and multiple SQL queries.
        * Manage optional read-through/write-back caching via a cache backend.
        * Reflect database schema metadata and expose table information helpers.

    Subclasses must implement:
        * from_uri
        * execute_query
        * execute_multiple_query

    Attributes:
        engine (Engine): SQLAlchemy engine connected to the target database.
        cache_db (Optional[MySQLCache]): Optional cache backend for query result caching.
        logger (Optional[logging.Logger]): Optional logger; a default logger is created if not provided.
        timeout (Optional[int | float]): Default timeout in seconds for query execution.
        metadata (MetaData): SQLAlchemy MetaData object reflecting the database schema.
        engine_url (str): String representation of the database engine URL.
        save_in_cache (bool): Whether to persist results into the cache on cache misses.

    Properties:
        dialect (str): The SQL dialect of the connected database (e.g., 'mysql', 'sqlite').
        db_id (str): Identifier for the database, derived from the engine URL.
        table_names (list[str]): List of table names in the connected database.
        inspector: SQLAlchemy Inspector object for introspecting the database schema.

    """

    def __init__(self,
                 engine: Engine,
                 relative_base_path: Path | str,
                 cache_db: Optional["BaseSQLDBExecutor"] = None,
                 logger: Optional[logging.Logger] = None,
                 timeout: Optional[int | float] = 400,
                 save_in_cache=False,
                 path_for_bm25_index: str = '.nl2sql_evaluator_cache/bm25_index',
                 path_tables_info_json: Optional[str] = None,
                 *args,
                 **kwargs):
        self.engine = engine
        self.relative_base_path = relative_base_path if isinstance(relative_base_path, Path) else Path(
            relative_base_path)
        self.cache_db = cache_db
        self.logger = logger or get_logger(name=__name__, level="INFO")
        self.timeout = timeout
        self.metadata = MetaData()
        self._reflect()
        self.engine_url = str(engine.url)
        if not self.table_names:
            self.logger.warning(f"No tables found in database at {self.engine_url}.")
        self.save_in_cache = save_in_cache
        self.path_for_bm25_index = os.path.join(path_for_bm25_index, self.db_id) if path_for_bm25_index else None
        self._index_db = None
        self.path_tables_info_json = Path(
            path_tables_info_json) if path_tables_info_json else self.relative_base_path.parent.parent.parent / "tables.json"
        if self.path_tables_info_json and not self.path_tables_info_json.suffix == ".json":
            raise ValueError(f"path_tables_info_json should be a json file. Got {self.path_tables_info_json}")
        self.tbl2col2descr = self._prepare_schema_filter_data()

        # during Python interpreter shutdown, objects can be torn down in arbitrary order. If a callback later tried to access self.engine, self (or its attributes) might already be partially destroyed.
        eng = self.engine  # capture by value
        # Registers a function to run right before the interpreter exits.
        atexit.register(lambda e=eng: e.dispose() if e is not None else None)

    # -----------------
    # Properties
    # -----------------
    @property
    def dialect(self) -> str:
        """Return the SQLAlchemy dialect name, ex. `mysql` or `sqlite`."""
        return self.engine.dialect.name

    @property
    def db_id(self) -> str:
        """
        Return an identifier for the database, derived from the engine URL.

        Raises:
            ValueError: If the dialect is unsupported.
        """
        if self.dialect == "mysql":
            return str(self.engine.url).strip("/")[-1]
        elif self.dialect == "sqlite":
            return str(self.engine.url).split("/")[-1].split("?")[0].split(".")[0]
        else:
            raise ValueError(
                f"Unsupported dialect: {self.dialect}. Cannot determine db_id."
            )

    @property
    def table_names(self) -> list[str]:
        """Return a list of table names in the database."""
        with self.engine.connect() as conn:
            return inspect(conn).get_table_names()

    @property
    def inspector(self):
        """Return a fresh SQLAlchemy inspector for the database bound to a fresh connection."""
        with self.engine.connect() as conn:
            output = inspect(conn)
        return output

    # -----------------
    # Abstract Methods
    # -----------------

    @classmethod
    @abstractmethod
    def from_uri(cls, *args, **kwargs) -> "BaseSQLDBExecutor":
        """
        Create an executor from a connection URI.

        See:
            https://docs.sqlalchemy.org/en/20/core/engines.html

        Returns:
            BaseSQLDBExecutor: Instance of a concrete executor.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    @abstractmethod
    def execute_query(
            self,
            query: str | sql.Executable,
            params: Optional[list[tuple]] = None,
            throw_if_error: bool = False,
            *args, **kwargs) -> list[tuple]:
        """
        Execute a single query against the database.

        Args:
            query (str | sql.Executable): SQL string or SQLAlchemy executable.
            params (Optional[list[tuple]]): Optional list of parameters for the query.
            throw_if_error (bool): If True, propagate exceptions; otherwise, handle/log internally.
            *args: Reserved for subclass extensions.
            **kwargs: Reserved for subclass extensions.

        Returns:
            list[tuple]: Result rows.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    @abstractmethod
    def execute_multiple_query(
            self,
            queries: list[str | sql.Executable],
            params: Optional[dict] = None,
            throw_if_error: bool = False,
            *args, **kwargs) -> list[list[tuple]]:
        """
        Execute multiple queries, possibly concurrently or in a transaction.

        Args:
            queries (list[str | sql.Executable]): List of SQL strings or SQLAlchemy executables.
            params (Optional[dict]): Optional parameters mapping; structure is implementation-defined.
            throw_if_error (bool): If True, propagate exceptions; otherwise, handle/log internally.
            *args: Reserved for subclass extensions.
            **kwargs: Reserved for subclass extensions.

        Returns:
            list[list[tuple]]: One result set (list of rows) per query.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    # -----------------
    # Execute query and cache
    # -----------------
    def execute_query_with_cache(self,
                                 query: str | sql.Executable,
                                 params: Optional[list[tuple]] = None,
                                 throw_if_error: bool = False,
                                 *args, **kwargs) -> list[tuple]:
        """
        Execute a query with optional read-through cache.

        If a cache is present and contains the result, returns the cached value.
        Otherwise, executes the query and optionally saves the result into the cache.

        Args:
            query (str | sql.Executable): SQL string or SQLAlchemy executable.
            params (Optional[list[tuple]]): Optional parameter list for the query.
            throw_if_error (bool): If True, re-raise execution exceptions.
            *args: Reserved for subclass extensions.
            **kwargs: Reserved for subclass extensions.

        Returns:
            list[tuple]: Result rows.
        """
        if self.cache_db is None:
            self.logger.debug(
                "Cache database is not set. Cannot execute query with caching. Executing without cache."
            )
            return self.execute_query(query, params, throw_if_error=throw_if_error, *args, **kwargs)

        # Check if the query result is already cached
        cached_result = self.cache_db.fetch_from_cache(self.db_id, str(query))
        if cached_result is not None:
            return cached_result

        self.logger.debug("Query not found in cache, executing query.")
        result = self.execute_query(query, params, throw_if_error=throw_if_error, *args, **kwargs)
        if self.save_in_cache:
            self.cache_db.insert_in_cache(self.db_id, str(query), result)
            self.logger.debug("Query cached in cache database.")
        return result

    # -----------------
    # Schema Reflection and Table Info
    # -----------------
    def _reflect(self):
        with self.engine.connect() as conn:
            self.metadata.reflect(bind=conn)
        return self.metadata

    def _prepare_schema_filter_data(self) -> dict[str, dict[str, str]]:
        if self.path_tables_info_json is None or not self.path_tables_info_json.exists():
            return {}

        db_info = pd.read_json(self.path_tables_info_json)
        db_info = db_info[db_info.db_id == self.db_id]
        if db_info.empty:
            self.logger.warning(f"No table info found for db_id {self.db_id} in {self.path_tables_info_json}")
            return {}

        tbl2col2descr = {}
        table_names_original = db_info["table_names_original"].values[0]
        column_names_original = db_info["column_names_original"].values[0]
        column_names = db_info["column_names"].values[0]
        for outer_table_idx, table_name_original in enumerate(table_names_original):
            tbl2col2descr[table_name_original] = {}
            for (inner_table_idx, column_name_original), (_, column_comment) in zip(column_names_original,
                                                                                    column_names):
                if outer_table_idx == inner_table_idx and column_name_original != column_comment:
                    tbl2col2descr[table_name_original][column_name_original] = column_comment

        return tbl2col2descr

    # -----------------
    # Table Info Helpers
    # -----------------
    def get_ddl_database(
            self,
            table_names: Optional[list[str]] = None,
            add_sample_rows_strategy: Optional[Literal["append", "inline"]] = None,
            question: Optional[str] = None,
    ) -> str:
        """Build a string with DDL for specified tables and optional sample rows.

        Args:
            table_names (Optional[list[str]]): Subset of table names to include; defaults to all tables.
            add_sample_rows_strategy (Optional[Literal["append", "inline"]]): Strategy to include sample rows.
                * `"inline"`: Append example values as comments next to columns.
                * `"append"`: Append INSERT statements with sampled rows.
                * `None`: Do not include sample data.
            question: Optional question string to guide sample data selection.

        Returns:
            str: Formatted DDL sections with optional sample data.

        Raises:
            ValueError: If any requested table is not present in the database.
        """
        table_in_db_lower = {name.lower() for name in self.table_names}
        table_names_lower = (
            {name.lower() for name in table_names} if table_names else table_in_db_lower
        )
        if table_names_lower.difference(table_in_db_lower):
            raise ValueError(
                f"Table names {table_names} not found in database. Available tables: {self.table_names}"
            )

        tables = []
        meta_tables = [
            tbl
            for tbl in self.metadata.sorted_tables
            if tbl.name.lower() in table_names_lower
               and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        for table in meta_tables:
            tbl_index = None
            if question:
                if table.name in self.index_db:
                    tbl_index = self.index_db[table.name]

            col2values_sim_quest = {} if question else None
            for k, v in table.columns.items():
                if type(v.type) is NullType:
                    table._columns.remove(v)
                if tbl_index and k in tbl_index:
                    col2values_sim_quest[k] = retrieve_from_bm25_index(
                        query=question,
                        retriever=tbl_index[k],
                        top_k=3
                    )

            with self.engine.connect() as conn:
                create_table = str(
                    CreateTable(table).compile(dialect=conn.dialect)
                )
                table_info = f"{create_table.rstrip()}"

                table_info = utils_augment_ddl_tbl(
                    ddl=table_info,
                    table=table,
                    execute_fn=self.execute_query,
                    dialect=conn.dialect,
                    strategy=add_sample_rows_strategy,
                    num_rows=2,
                    col2values_sim_quest=col2values_sim_quest,
                    col2descr=self.tbl2col2descr.get(table.name, {}),
                )

                tables.append(table_info)

        tables.sort()
        final_str = "\n\n".join(tables)
        return final_str

    @property
    def index_db(self) -> dict[str, dict[str, Any]]:
        """
        Build BM25 retrievers for categorical (string-ish) columns.
        Skips SQLite system tables, NULL/empty strings, and numeric-looking values.
        """
        if self._index_db is not None:
            return self._index_db

        self.logger.info("Building BM25 indexes for categorical columns")

        # Filter out SQLite internal tables

        meta_tables = [
            tbl for tbl in self.metadata.sorted_tables
            if not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables2indexes: dict[str, dict[str, Any]] = {}

        # Treat only text columns as candidates (String/Text and variants)
        stringish = (String, Text)

        for table in meta_tables:
            col_indexes: dict[str, Any] = {}

            for col in table.columns:
                # Only index string-like columns
                if not isinstance(getattr(col, "type", None), stringish):
                    continue

                # DISTINCT normalized values directly in SQL where possible
                # - Trim and guard against NULL/empty
                # - Left-substring to 40 chars to bound memory early
                # Some dialects lack SUBSTR/LTRIM; fall back to Python if needed.
                substr = sa.func.substr(col, 1, 40)
                trimmed = sa.func.trim(substr)
                stmt = (
                    sa.select(sa.func.distinct(trimmed))
                    .where(sa.and_(col.is_not(None), trimmed != ""))
                    .limit(5000)
                )

                try:
                    # Expect rows like [(value,), (value,), ...]
                    rows = self.execute_query(stmt)
                    values = [r[0] for r in rows if r and r[0] is not None]
                except Exception:
                    # Fallback path if the SQL functions aren’t supported by the dialect
                    rows = self.execute_query(sa.select(col).distinct().where(col.is_not(None)).limit(5000))
                    values = [str(r[0])[:40].strip() for r in rows if
                              r and r[0] is not None and str(r[0]).strip() != ""]

                # Ensure index path exists
                index_path = os.path.join(self.path_for_bm25_index, table.name, col.name)
                # Build the retriever
                retriever = create_bm25_index(values, index_path)
                if retriever:
                    col_indexes[col.name] = retriever

            if col_indexes:
                tables2indexes[table.name] = col_indexes

        self._index_db = tables2indexes
        return self._index_db

    def retrieve_similar_col_values_from_db_index(self, query, column_name, table_name, top_k=2):
        """
        Retrieve similar column values from the BM25 index for a given query.

        Args:
            query (str): The input query string to search for similar values.
            column_name (str): The name of the column to search within.
            table_name (str): The name of the table containing the column.
            top_k (int): The number of top similar values to retrieve.

        Returns:
            list[str]: A list of similar column values.
        """
        retriever = self.index_db[table_name][column_name]
        docs = retriever.retrieve(bm25s.tokenize(query), k=top_k)
        docs = [doc['text'] if isinstance(doc, dict) else doc for doc in docs[0]]
        return docs

    def dispose(self):
        """Dispose of the engine and clean up connections."""
        try:
            if getattr(self, "engine", None) is not None:
                self.engine.dispose()
        except Exception:
            # swallow on shutdown
            pass
