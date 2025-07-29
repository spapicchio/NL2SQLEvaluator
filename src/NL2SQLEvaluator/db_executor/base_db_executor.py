import logging
import re
from abc import ABC, abstractmethod
from sqlite3 import ProgrammingError
from typing import Optional, Literal

from langgraph.func import task
from sqlalchemy import Engine, inspect, Table, insert, select, MetaData
from sqlalchemy import sql
from sqlalchemy.sql.ddl import CreateTable
from sqlalchemy.sql.sqltypes import NullType

from NL2SQLEvaluator.db_executed_cache import MySQLCache
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.task_definition import SingleTask, SQLTask


@task()
def db_executor_worker(single_task: SingleTask) -> SingleTask:
    """
    Worker function to execute SQL queries in a batch.
    """
    engine = single_task.engine

    executed_target = engine.execute_query_and_cache(single_task.target_sql.query) \
        if single_task.target_sql.executed is None else single_task.target_sql.executed

    executed_predicted = engine.execute_query_and_cache(single_task.predicted_sql.query) \
        if single_task.predicted_sql.executed is None else single_task.predicted_sql.executed

    target = SQLTask(query=single_task.target_sql.query, executed=executed_target)
    predicted = SQLTask(query=single_task.predicted_sql.query, executed=executed_predicted)

    return SingleTask(
        target_sql=target,
        predicted_sql=predicted,
        **single_task.model_dump(exclude={"target_sql", "predicted_sql"})
    )


class BaseSQLDBExecutor(ABC):
    def __init__(self,
                 engine: Engine,
                 cache_db: Optional[MySQLCache] = None,
                 logger: Optional[logging.Logger] = None,
                 timeout: Optional[int | float] = 400,
                 save_in_cache=False,
                 *args,
                 **kwargs):
        self.timeout = timeout
        self.engine = engine
        self.cache_db = cache_db
        self.logger = logger or get_logger(name=__name__, level="INFO")
        self.metadata = MetaData()
        self._reflect()
        self.engine_url = str(engine.url)
        if not self.table_names:
            self.logger.error(f"No tables found in database at {self.engine_url}.")
        self.save_in_cache = save_in_cache

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self.engine.dialect.name

    @property
    def db_id(self) -> str:
        if self.dialect == "mysql":
            return str(self.engine.url).strip("/")[-1]
        elif self.dialect == "sqlite":
            return str(self.engine.url).split("/")[-1].split("?")[0].split(".")[0]
        else:
            raise ValueError(
                f"Unsupported dialect: {self.dialect}. Cannot determine db_id."
            )

    @classmethod
    @abstractmethod
    def from_uri(cls, *args, **kwargs) -> "BaseSQLDBExecutor":
        # https://docs.sqlalchemy.org/en/20/core/engines.html
        raise NotImplementedError("This method should be implemented by subclasses")

    def execute_query_and_cache(self,
                                query: str | sql.Executable,
                                params: Optional[list[tuple]] = None,
                                throw_if_error: bool = False,
                                *args, **kwargs) -> list[tuple]:
        if self.cache_db is None:
            self.logger.debug(
                "Cache database is not set. Cannot execute query with caching. Executing without cache."
            )
            return self.execute_query(query, params, throw_if_error=throw_if_error, *args, **kwargs)

        # Check if the query result is already cached
        cached_result = self.cache_db.get_from_cache(self.db_id, str(query))
        if cached_result is not None:
            return cached_result

        self.logger.debug("Query not found in cache, executing query.")
        result = self.execute_query(query, params, throw_if_error=throw_if_error, *args, **kwargs)
        if self.save_in_cache:
            self.cache_db.insert_in_cache(self.db_id, str(query), result)
            self.logger.debug("Query cached in cache database.")
        return result

    @abstractmethod
    def execute_query(
            self,
            query: str | sql.Executable,
            params: Optional[list[tuple]] = None,
            throw_if_error: bool = False,
            *args, **kwargs) -> list[tuple]:
        raise NotImplementedError("This method should be implemented by subclasses")

    @abstractmethod
    def execute_multiple_query(
            self,
            queries: list[str | sql.Executable],
            params: Optional[dict] = None,
            throw_if_error: bool = False,
            *args, **kwargs) -> list[list[tuple]]:
        """Execute multiple queries in a single transaction."""
        raise NotImplementedError("This method should be implemented by subclasses")

    @property
    def table_names(self) -> list[str]:
        """Return a list of table names in the database."""
        with self.engine.connect() as conn:
            return inspect(conn).get_table_names()

    def _reflect(self):
        with self.engine.connect() as conn:
            self.metadata.reflect(bind=conn)
        return self.metadata

    @property
    def inspector(self):
        """Get the SQLAlchemy inspector for the database."""
        with self.engine.connect() as conn:
            output = inspect(conn)
        return output

    def get_table_info(
            self,
            table_names: Optional[list[str]] = None,
            add_sample_rows_strategy: Optional[Literal["append", "inline"]] = None,
    ) -> str:
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
            if tbl.name in table_names_lower
               and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        for table in meta_tables:
            for k, v in table.columns.items():
                if type(v.type) is NullType:
                    table._columns.remove(v)
            with self.engine.connect() as conn:
                create_table = str(
                    CreateTable(table).compile(dialect=conn.dialect)
                )
                table_info = f"{create_table.rstrip()}"

                if add_sample_rows_strategy and add_sample_rows_strategy == "inline":
                    table_info = self._add_inline_example_rows(
                        table_info, table, num_rows=5
                    )
                elif add_sample_rows_strategy and add_sample_rows_strategy == "append":
                    insert_into = self._return_insert_rows_dump(table, num_rows=5)
                    table_info = f"{table_info}\n{insert_into}"

                tables.append(table_info)

        tables.sort()
        final_str = "\n\n".join(tables)
        return final_str

    def _return_select_rows(self, table: Table, num_rows: int = 5):
        """Return select rows using thread-safe session."""
        command = select(table).limit(num_rows)
        sample_rows_result = []
        try:
            sample_rows_result = self.execute_query(
                query=command, mysql_cache=None
            )
        except ProgrammingError as e:
            self.logger.error(f"Error executing select query on table {table.name}: {e}. Skipping sample rows.")

        return sample_rows_result

    def _add_inline_example_rows(self, table_info, table, num_rows=5):
        sample_rows = list(self._return_select_rows(table, num_rows))
        col_examples = {}
        if sample_rows:
            for idx, col in enumerate(table.columns):
                examples = {row[idx] for row in sample_rows if row[idx] is not None}
                if examples:
                    col_examples[col.name] = f"Example Values: {tuple(examples)}"

        lines = table_info.splitlines()
        new_lines = []
        for line in lines:
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

    def _return_insert_rows_dump(self, table: Table, num_rows: int = 5) -> str:
        sample_rows_result = self._return_select_rows(table, num_rows)
        inserts = []

        with self.engine.connect() as conn:
            for row in sample_rows_result:
                row = [str(i)[:100] for i in row]
                stmt = insert(table).values(dict(zip(table.columns.keys(), row)))
                compiled = stmt.compile(
                    dialect=conn.dialect, compile_kwargs={"literal_binds": True}
                )
                inserts.append(str(compiled))

        return "\n".join(inserts)
