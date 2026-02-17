"""Module for caching SQL execution results using SQLite.

This module provides a persistent SQLite-backed cache to store and retrieve 
SQLOutputTable objects based on a unique hash of the query and database context.
"""

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import (
    NotFoundInCacheError, DataToFetch, DataToCache
)
from NL2SQLEvaluator.db_executor_nodes.db_executor_input import ExecutorError, TaskToBeExecuted
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SQLExecutorOutput
from NL2SQLEvaluator.db_executor_nodes.sqlite_db_executor import SQLiteDBExecutor
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import register_node

logger = get_logger(__name__)


@register_node()
class SqliteCache:
    """
    Persistent cache provider using SQLite.

    Why: Reduces latency and computational cost for repeated queries by
    storing compressed results.
    How: Initialize with a database path; it uses an internal SQLiteDBExecutor
    to manage the cache table and perform I/O.
    """

    def __init__(self, cache_db_path: str):
        """Initializes the executor and ensures the cache table exists."""
        self.executor = SQLiteDBExecutor()
        self.cache_db_path = cache_db_path
        self._init_db()

    def _init_db(self) -> None:
        """Creates the cache table schema if it doesn't exist."""
        sql = """
              CREATE TABLE IF NOT EXISTS `cache_data` \
              ( \
                  `hash_key` TEXT PRIMARY KEY, \
                  `db_path`  TEXT NOT NULL, \
                  `query`    TEXT NOT NULL, \
                  `result`   BLOB NOT NULL
              ); \
              """
        task = TaskToBeExecuted(db_path=self.cache_db_path, queries=[sql])
        self.executor.execute_queries([task], allow_write=True)

    def set_in_cache(self, data_to_cache: list[DataToCache]) -> None:
        """Persists a list of execution results to the cache.

        Args:
            data_to_cache: Objects containing the hash and SQLOutputTable.
        """
        insert_sql = """
                     INSERT OR IGNORE INTO `cache_data` (hash_key, db_path, query, result)
                     VALUES (:hash_key, :db_path, :query, :result); \
                     """
        params = [
            {
                **data.model_dump(exclude={'result', 'dialect'}),
                'result': data.result.compress()
            }
            for data in data_to_cache
        ]

        task = TaskToBeExecuted(
            db_path=self.cache_db_path,
            queries=[insert_sql] * len(params),
            params=params,
        )

        execution_results = self.executor.execute_queries([task], allow_write=True)

        if any(isinstance(res, ExecutorError) for res in execution_results[0]):
            logger.error(f"Failed to write to cache: {execution_results[0]}")

    def get_from_cache(
            self,
            data_to_fetch: list[DataToFetch]
    ) -> list[SQLExecutorOutput | NotFoundInCacheError]:
        """Retrieves results from the cache based on hash_key.

        Args:
            data_to_fetch: Metadata objects containing the hash_key.

        Returns:
            A list where each index corresponds to the input list, containing
            either the decompressed table or a NotFoundInCacheError.
        """
        select_sql = "SELECT result FROM `cache_data` WHERE hash_key = :hash_key;"
        params = [{"hash_key": d.hash_key} for d in data_to_fetch]

        task = TaskToBeExecuted(
            db_path=self.cache_db_path,
            queries=[select_sql] * len(params),
            params=params,
        )

        # Results from execute_queries are nested: [TaskIdx][QueryIdx]
        batch_results = self.executor.execute_queries([task], allow_write=False)[0]
        final_output = [
            NotFoundInCacheError(str(ExecutorError))
            # if the hash key is not present in the database, the query will return an empty list of rows
            if isinstance(result_rows, ExecutorError) or result_rows.rows == []
            else SQLExecutorOutput.decompress(result_rows.rows[0][0])
            for result_rows in batch_results
        ]
        return final_output
