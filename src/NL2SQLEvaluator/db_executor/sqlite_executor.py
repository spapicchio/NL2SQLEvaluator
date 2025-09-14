import concurrent
import contextlib
import hashlib
import os
import pickle
import sqlite3
import time
from typing import Optional

import sqlglot
from func_timeout import func_set_timeout, FunctionTimedOut
from sqlalchemy import create_engine, text, sql, event, Engine
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm
from typing_extensions import override

from NL2SQLEvaluator.db_executor.base_db_executor import BaseSQLDBExecutor
from NL2SQLEvaluator.logger import get_logger


class SqliteDBExecutor(BaseSQLDBExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._install_pragmas_listener()

    def set_different_timeout(self, timeout: int | float) -> None:
        """Set a different timeout for the current instance."""
        self.timeout = timeout

    @contextlib.contextmanager
    def connection(self):
        with self.engine.connect() as con:
            yield con

    @classmethod
    def from_uri(cls, *args, **kwargs) -> "SqliteDBExecutor":
        logger = get_logger(name=__name__, level="INFO")
        db_path = kwargs.get("relative_base_path")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite database file not found at {db_path}")
        uri = f"sqlite:///{db_path}?mode=ro&nolock=1&check_same_thread=false&immutable=1"
        # uri = f"sqlite:///{db_path}?check_same_thread=false&immutable=1"
        logger.info(f"Connecting to SQLite database with URI {uri}")
        engine = create_engine(uri,
                               # echo=True, echo_pool=True,
                               pool_size=20, max_overflow=40,
                               pool_timeout=60,
                               connect_args={
                                   "timeout": 60
                               },
                               pool_pre_ping=True,
                               pool_recycle=1800)
        logger.warning(f"Created ENGINE for high concurrency read settings but NO WRITE.")
        return cls(engine=engine, cache_db=None)

    def _install_pragmas_listener(self) -> None:
        """Attach a connect-time callback to *this* engine instance."""

        @event.listens_for(Engine, "connect")
        def _sqlite_on_connect(dbapi_connection, _):
            if not isinstance(dbapi_connection, sqlite3.Connection):
                return  # Ignore non‑SQLite engines

            cursor = dbapi_connection.cursor()
            try:
                # PRAGMA journal_mode=WAL;
                # PRAGMA synchronous=NORMAL;
                # PRAGMA temp_store=MEMORY;
                cursor.executescript("""
                    PRAGMA foreign_keys=ON;
                    PRAGMA mmap_size=30000000000;
                """)
                self.logger.debug("Installed SQLite PRAGMAs for high concurrency reads.")
            finally:
                cursor.close()

    def execute_query(self,
                      query: str | sql.Executable,
                      params: Optional[dict] = None,
                      throw_if_error: bool = False,
                      *args, **kwargs) -> Optional[list[tuple]]:

        @func_set_timeout(self.timeout)
        def _execute_with_timeout(query, params):
            query = text(query) if isinstance(query, str) else query
            is_write = False
            if 'INSERT' in str(query).upper() or 'UPDATE' in str(query).upper() or 'DELETE' in str(
                    query).upper() or 'CREATE' in str(query).upper() or 'DROP' in str(query).upper() or 'ALTER' in str(
                query).upper():
                is_write = True
            try:
                with self.connection() as conn:
                    if is_write:
                        conn.execute(query, params)
                        conn.commit()
                        result = []
                    else:
                        cursor = conn.execute(query, params)
                        rows = cursor.fetchall()
                        result = [row._tuple() for row in rows]

            except SQLAlchemyError as e:
                self.logger.warning(e)
                if throw_if_error:
                    raise e
                result = None
            return result

        try:
            return _execute_with_timeout(query, params)
        except FunctionTimedOut as e:
            self.logger.warning(f"execute_query timed out after {self.timeout}. Returning None.")
            return None

    def execute_multiple_query(self,
                               queries: list[str | sql.Executable],
                               params: Optional[list[dict]] = None,
                               throw_if_error: bool = False,
                               max_thread_num: int = 50,
                               *args,
                               **kwargs) -> list[list[tuple] | None]:
        if len(queries) == 1:
            return [self.execute_query_with_cache(queries[0], params[0] if params else None)]
        self.logger.debug(
            f"Executing multiple {len(queries)} queries concurrently with max_thread_num={max_thread_num} and timeout={self.timeout}"
        )
        self.logger.debug(
            f"Number of queries={len(queries)}"
        )
        start = time.time()
        params = params or [None] * len(queries)
        results = [None] * len(queries)
        num_thread = min(len(queries), max_thread_num)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_thread) as executor:
            futures = {
                executor.submit(self.execute_query_with_cache, q, p, throw_if_error=throw_if_error): i
                for i, (q, p) in enumerate(zip(queries, params))
            }
            for future in tqdm(concurrent.futures.as_completed(futures), desc="Executing query"):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    self.logger.warning(f'Query generated an exception: {exc}')
                    results[idx] = None
        self.logger.debug(
            f"Executed multiple queries in {time.time() - start:.2f} seconds",
        )
        return results


def hash_db_id_sql(db_id, query) -> str:
    value = f"{db_id}|{query}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class SqliteCacheDB(SqliteDBExecutor):
    @override
    def _install_pragmas_listener(self) -> None:
        """Attach a connect-time callback to *this* engine instance."""

        @event.listens_for(Engine, "connect")
        def _sqlite_on_connect(dbapi_connection, _):
            if not isinstance(dbapi_connection, sqlite3.Connection):
                return  # Ignore non‑SQLite engines

            cursor = dbapi_connection.cursor()
            try:
                # PRAGMA journal_mode=WAL;
                # PRAGMA synchronous=NORMAL;
                # PRAGMA temp_store=MEMORY;
                cursor.executescript("""
                    PRAGMA journal_mode=WAL;
                    PRAGMA synchronous=NORMAL;
                    PRAGMA foreign_keys=ON;
                    PRAGMA mmap_size=30000000000;
                """)
                self.logger.debug("Installed SQLite PRAGMAs for high concurrency reads.")
            finally:
                cursor.close()

    @classmethod
    @override
    def from_uri(cls, *args, **kwargs) -> "SqliteDBExecutor":
        logger = get_logger(name=__name__, level="INFO")
        db_path = kwargs.get("relative_base_path")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite database file not found at {db_path}")
        uri = f"sqlite:///{db_path}?mode=rwc&cache=shared&check_same_thread=false"
        logger.info(f"Connecting to SQLite database with URI {uri}")
        engine = create_engine(uri,
                               # echo=True, echo_pool=True,
                               pool_size=20, max_overflow=40,
                               pool_timeout=60,
                               connect_args={
                                   "timeout": 60
                               },
                               pool_pre_ping=True,
                               pool_recycle=1800)
        initiated_object = cls(engine=engine, cache_db=None)
        initiated_object.create_cache_table()
        return initiated_object

    def db_id_query_already_present(self) -> set[tuple]:
        stm = text("SELECT `db_id`, `query` FROM cache_data")
        return set(self.execute_query(stm))

    def create_cache_table(self) -> None:
        """Create the cache table if it does not exist."""
        create_table_sql = """
                           CREATE TABLE IF NOT EXISTS `cache_data`
                           (
                               `hash_key` TEXT PRIMARY KEY,
                               `db_id`    TEXT NOT NULL,
                               `query`    TEXT NOT NULL,
                               `result`   BLOB NOT NULL
                           );
                           """.strip()
        self.execute_query(create_table_sql)

    def insert_in_cache(self, db_id: str, query: str, result: list[tuple]) -> None:
        query = self.parse_sql_query(query)
        hash_id = hash_db_id_sql(db_id, query)
        self.logger.debug(
            f"Inserting (or Ignore if duplicate) in cache with hash_id: {hash_id}"
        )
        pickled_result = pickle.dumps(result)
        hash_id = hash_db_id_sql(db_id, query)
        stmt = "INSERT OR IGNORE INTO `cache_data` (hash_key, db_id, query, result) VALUES (:hash_id, :db_id, :query, :result)"
        try:
            self.execute_query(stmt, {"hash_id": hash_id, "db_id": db_id, "query": query, "result": pickled_result})
        except Exception as e:
            self.logger.error(
                f"Failed to insert into cache for `{db_id}`, `{query}`, error: {e}"
            )
        return None

    def insert_bulk_in_cache(self, db_ids: list[str], queries: list[str], results: list[list[tuple]]) -> None:
        if len(queries) != len(results):
            self.logger.error("Length of queries and results must be the same.")
            return

        self.logger.debug(f"Inserting {len(queries)} entries in bulk into cache.")
        to_insert = []
        for db_id, query, result in zip(db_ids, queries, results):
            parsed_query = self.parse_sql_query(query)
            hash_id = hash_db_id_sql(db_id, parsed_query)
            pickled_result = pickle.dumps(result)
            to_insert.append({"hash_id": hash_id, "db_id": db_id, "query": parsed_query, "result": pickled_result})
        stmt = "INSERT OR IGNORE INTO `cache_data` (hash_key, db_id, query, result) VALUES (:hash_id, :db_id, :query, :result)"
        try:
            with self.connection() as conn:
                conn.execute(text(stmt), to_insert)
                conn.commit()
        except Exception as e:
            self.logger.error(
                f"Failed to insert bulk into cache error: {e}"
            )
            self.logger.warning('Bulk failed to insert, trying one by one.')
            for query, result in zip(queries, results):
                self.insert_in_cache(db_id, query, result)
        return None

    def fetch_from_cache(self, db_id: str, query: str) -> Optional[list[tuple]]:
        """Fetch the result of a query from the cache."""
        query = self.parse_sql_query(query)
        hash_id = hash_db_id_sql(db_id, query)
        self.logger.debug(f"Fetching from cache with hash_id: {hash_id}")

        try:
            stmt = "SELECT `result` FROM `cache_data` WHERE hash_key = :hash_id"
            result_row = self.execute_query(query=stmt, params={"hash_id": hash_id})
            return pickle.loads(result_row[0][0]) if result_row else None
        except Exception as e:
            self.logger.error(
                f"Failed to retrieve from cache for `{db_id}`, `{query}`, error: {e}"
            )
        return None

    def parse_sql_query(self, query: str, dialect: str = "sqlite") -> str:
        try:
            parsed_query = sqlglot.transpile(query, dialect, identity=True)
            return parsed_query[0]
        except Exception as e:
            self.logger.error(f"Failed to parse SQL query: {query}, error: {e}")
            return query
