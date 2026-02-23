"""Module for caching query execution results using SQLite.

This module provides a persistent SQLite-backed cache to store and retrieve
GenericExecutorOutput objects based on a unique hash of the query and database context.
"""

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import (
    NotFoundInCacheError, DataToFetch, DataToCache
)
from NL2SQLEvaluator.db_executor_nodes.db_executor_input import TaskToBeExecuted
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import (
    SQLExecutorOutput, CypherExecutorOutput, SparqlExecutorOutput,
    GenericExecutorOutput, ExecutorError,
)
from NL2SQLEvaluator.db_executor_nodes.sqlite_db_executor import SQLiteDBExecutor
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import register_node

logger = get_logger(__name__)

_DIALECT_TO_OUTPUT_CLASS: dict[str, type[GenericExecutorOutput]] = {
    "sql": SQLExecutorOutput,
    "sqlite": SQLExecutorOutput,
    "postgres": SQLExecutorOutput,
    "cypher": CypherExecutorOutput,
    "neo4j": CypherExecutorOutput,
    "sparql": SparqlExecutorOutput,
}


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

    def set_in_cache(self, cache_path: str, data_to_cache: list[DataToCache]) -> None:
        """Persists a list of execution results to the cache.

        Args:
            cache_path: Path to the cache database (unused, instance uses self.cache_db_path).
            data_to_cache: Objects containing the hash and execution result.
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
            cache_path: str,
            data_to_fetch: list[DataToFetch]
    ) -> list[DataToCache | NotFoundInCacheError]:
        """Retrieves results from the cache based on hash_key.

        Args:
            cache_path: Path to the cache database (unused, instance uses self.cache_db_path).
            data_to_fetch: Metadata objects containing the hash_key and dialect.

        Returns:
            A list where each index corresponds to the input list, containing
            either a DataToCache with the decompressed result or a NotFoundInCacheError.
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
        final_output: list[DataToCache | NotFoundInCacheError] = []
        for i, result_rows in enumerate(batch_results):
            if isinstance(result_rows, ExecutorError) or result_rows.rows == []:
                final_output.append(NotFoundInCacheError(f"Cache miss for hash {data_to_fetch[i].hash_key}"))
            else:
                output_class = _DIALECT_TO_OUTPUT_CLASS.get(data_to_fetch[i].dialect, SQLExecutorOutput)
                result = output_class.decompress(result_rows.rows[0][0])
                final_output.append(DataToCache(
                    db_path=data_to_fetch[i].db_path,
                    query=data_to_fetch[i].query,
                    dialect=data_to_fetch[i].dialect,
                    result=result,
                ))
        return final_output
