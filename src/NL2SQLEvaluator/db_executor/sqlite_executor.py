import concurrent
import sqlite3
import time
from typing import Optional

from sqlalchemy import create_engine, text, sql, event, Engine
from sqlalchemy.exc import SQLAlchemyError

from NL2SQLEvaluator.db_executor.base_db_executor import BaseSQLDBExecutor
from NL2SQLEvaluator.logger import get_logger


class SqliteDBExecutor(BaseSQLDBExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._install_pragmas_listener()

    @classmethod
    def from_uri(cls, *args, **kwargs) -> "SqliteDBExecutor":
        # TODO update cache DB
        logger = get_logger(name=__name__, level="DEBUG")
        db_path = kwargs.get("relative_base_path")
        uri = f"sqlite:///file:{db_path}?mode=ro&nolock=1&uri=true&check_same_thread=false"
        logger.info(f"Connecting to SQLite database with URI {uri}")
        engine = create_engine(uri,
                               # echo=True, echo_pool=True,
                               pool_size=20, max_overflow=40,
                               pool_timeout=60, connect_args={"timeout": 60},
                               pool_pre_ping=True,
                               pool_recycle=1800)
        logger.warning(f"Created ENGINE for high concurrency read settings but NO WRITE.")
        return cls(engine=engine, cache_db=None)

    def _install_pragmas_listener(self) -> None:
        """Attach a connect-time callback to *this* engine instance."""

        @event.listens_for(Engine, "connect")
        def _sqlite_on_connect(dbapi_connection, _):
            import sqlite3
            if not isinstance(dbapi_connection, sqlite3.Connection):
                return  # Ignore non‑SQLite engines

            cursor = dbapi_connection.cursor()
            try:
                cursor.executescript("""
                    PRAGMA foreign_keys=ON;
                    PRAGMA journal_mode=WAL;
                    PRAGMA synchronous=NORMAL;
                    PRAGMA temp_store=MEMORY;
                    PRAGMA mmap_size=30000000000;
                """)
                self.logger.debug("Installed SQLite PRAGMAs for high concurrency reads.")
            finally:
                cursor.close()

    def _install_timeout(self, engine, max_ms=1_000, vm_steps=1_000):
        """Abort any statement that runs longer than max_ms (wall‑clock)."""

        @event.listens_for(engine, "before_cursor_execute")
        def _before(conn, cursor, statement, params, context, executemany):
            start = time.perf_counter()

            def _progress():
                if (time.perf_counter() - start) * 1000 > max_ms:
                    return 1  # non‑zero → SQLITE_INTERRUPT
                return 0

            conn.connection.set_progress_handler(_progress, vm_steps)

        @event.listens_for(engine, "after_cursor_execute")
        def _after(conn, *_):
            # remove handler so the next statement starts with a clean slate
            conn.connection.set_progress_handler(None, 0)

    def execute_query(self,
                      query: str | sql.Executable,
                      params: Optional[dict] = None,
                      throw_if_error: bool = False,
                      timeout=400,
                      *args, **kwargs) -> Optional[list[tuple]]:

        self._install_timeout(self.engine, max_ms=timeout * 1000, vm_steps=1000)

        def _execute_and_fetch(conn_, query_, params_):
            cursor = conn_.execute(query_, params_)
            rows = cursor.fetchall()
            return [row._tuple() for row in rows]

        query = text(query) if isinstance(query, str) else query
        try:
            with self.engine.connect() as conn:
                # result = func_timeout(timeout, _execute_and_fetch, args=(conn, query, params or {}))
                result = _execute_and_fetch(conn, query, params)
        except SQLAlchemyError as e:
            if 'interrupted' in str(e):
                self.logger.warning(f"SQL Timeout after {timeout}: error: {e}")
            else:
                self.logger.warning(e)
            if throw_if_error:
                raise e
            result = None
        return result

    def execute_multiple_query(self, queries: list[str | sql.Executable],
                               params: Optional[list[dict]] = None,
                               throw_if_error: bool = False,
                               max_thread_num: int = 50,
                               timeout: int | float = 400,
                               *args,
                               **kwargs) -> list[list[tuple] | None]:

        self._install_timeout(self.engine, max_ms=timeout * 1000, vm_steps=1000)
        if len(queries) == 1:
            return [self.execute_query(queries[0], params[0] if params else None)]
        self.logger.debug(
            f"Executing multiple {len(queries)} queries concurrently with max_thread_num={max_thread_num} and timeout={timeout}"
        )
        self.logger.debug(
            f"Number of queries= {len(queries)}"
        )
        start = time.time()
        params = params or [None] * len(queries)
        results = [None] * len(queries)
        num_thread = min(len(queries), max_thread_num)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_thread) as executor:
            futures = {
                executor.submit(self.execute_query, q, p, throw_if_error=throw_if_error, timeout=timeout): i
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
