from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import NotFoundInCacheError, OutputTable, DataToFetch, \
    DataToCache
from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecutorError, ExecuteTask
from NL2SQLEvaluator.db_executor_nodes.sqlite_db_executor import SQLiteDBExecutor, _execute_single_query
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import register_node

logger = get_logger(__name__)

_TABLE_EXISTS = False


def create_cache_table(db_file) -> None:
    """Create the cache table if it does not exist."""
    create_table_sql = """
                       CREATE TABLE IF NOT EXISTS `cache_data`
                       (
                           `hash_key` TEXT PRIMARY KEY,
                           `db_id`    TEXT NOT NULL,
                           `query`    TEXT NOT NULL,
                           `result`   BLOB NOT NULL
                       ); \
                       """.strip()
    _execute_single_query(0, 0, db_file, create_table_sql, timeout=300, allow_write=True)
    _TABLE_EXISTS = True


@register_node()
class SqliteCache:
    @staticmethod
    def set_in_cache(db_file: str,
                     data_to_cache: list[DataToCache]) -> None:
        """Save a file with the given name and parameters."""
        # create the cache table if it does not exist:
        if not _TABLE_EXISTS:
            create_cache_table(db_file)
        insert_sql = """
                     INSERT OR IGNORE INTO `cache_data` (hash_key, db_id, query, result)
                     VALUES (:hash_key, :db_id, :query, :result);
                     """.strip()

        params = [
            data.model_dump(exclude={'result', 'dialect'}) | {'result': data.result.compress()}
            # This if is used to not store empty results which is used to detect if not found in cache
            if len(data.result) > 0 else data.model_dump(exclude={'result', 'dialect'}) | {
                'result': OutputTable(rows=[('empty',)]).compress()}
            for data in data_to_cache
        ]

        results = SQLiteDBExecutor.execute_queries(
            tasks=[ExecuteTask(
                db_files=db_file,
                queries=[insert_sql] * len(params),
                params=params,
            )],
            allow_write=True,
            timeout=3000,
        )[0][0]

        if isinstance(results, ExecutorError):
            logger.error(f'Impossible to set cache in the database. error: {results}')

    @staticmethod
    def get_from_cache(db_file: str, data_to_fetch: list[DataToFetch]) -> list[OutputTable | NotFoundInCacheError]:
        """Retrieve a file with the given name and parameters."""

        if not _TABLE_EXISTS:
            create_cache_table(db_file)

        select_sql = """
                     SELECT result
                     FROM `cache_data`
                     WHERE hash_key = :hash_key;
                     """.strip()
        params = [data.model_dump(include={'hash_key'}) for data in data_to_fetch]

        tasks = [ExecuteTask(
            db_files=db_file,
            queries=[select_sql] * len(params),
            params=params,
        )]
        results = SQLiteDBExecutor.execute_queries(
            tasks=tasks,
            allow_write=False,
            timeout=30,
        )[0]
        dec_results = []
        for data in results:
            decompressed = data.decompress()
            if len(decompressed) == 0:
                dec_results.append(NotFoundInCacheError())
            elif decompressed.rows == [('empty',)]:
                dec_results.append(OutputTable(rows=[]))
            else:
                dec_results.append(decompressed)
        return dec_results
