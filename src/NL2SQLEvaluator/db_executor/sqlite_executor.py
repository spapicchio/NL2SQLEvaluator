import concurrent
import contextlib
import os
import sqlite3
import time
from typing import Optional

from func_timeout import func_set_timeout, FunctionTimedOut
from sqlalchemy import create_engine, text, sql, event, Engine
from sqlalchemy.exc import SQLAlchemyError

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
        # TODO update cache DB
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
            try:
                with self.connection() as conn:
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
            return [self.execute_query_and_cache(queries[0], params[0] if params else None)]
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
                executor.submit(self.execute_query_and_cache, q, p, throw_if_error=throw_if_error): i
                for i, (q, p) in enumerate(zip(queries, params))
            }
            for future in concurrent.futures.as_completed(futures):
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
