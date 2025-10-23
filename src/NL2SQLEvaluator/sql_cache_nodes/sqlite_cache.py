import hashlib
import pickle

import sqlglot
from NL2SQLEvaluator.db_executors_nodes.db_executor_protocol import OutputTable, ExecutorError
from NL2SQLEvaluator.db_executors_nodes.sqlite_db_executor import SQLiteDBReader
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import register_node
from NL2SQLEvaluator.sql_cache_nodes.sql_cache_protocol import NotFoundInCacheError

logger = get_logger(__name__)


def parse_sql_query(query: str, dialect: str = "sqlite") -> str:
    try:
        parsed_query = sqlglot.transpile(query, dialect, identity=True)
        return parsed_query[0]
    except Exception as e:
        logger.error(f"Failed to parse SQL query: {query}, error: {e}")
        return query


def hash_db_id_sql(db_id, query) -> str:
    value = f"{db_id}|{query}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    SQLiteDBReader._execute_single_query(db_file, create_table_sql, timeout_s=300, allow_write=True)


def compress_data(data) -> bytes:
    return pickle.dumps(data)


def uncompress_data(data) -> OutputTable | NotFoundInCacheError:
    return pickle.loads(data) if not isinstance(data, Exception) else NotFoundInCacheError(data)


@register_node()
class SqliteCache:
    @staticmethod
    def set_in_cache(db_file: str,
                     db_ids: list[str],
                     queries: list[str],
                     executed_queries: list[OutputTable]) -> None:
        """Save a file with the given name and parameters."""
        # create the cache table if it does not exist:
        create_cache_table(db_file)
        insert_sql = """
                     INSERT OR IGNORE INTO `cache_data` (hash_key, db_id, query, result)
                     VALUES (?, ?, ?, ?);
                     """.strip()

        params = [
            {
                'hash_key': hash_db_id_sql(db_id, parse_sql_query(query)),
                'db_id': db_id,
                'query': parse_sql_query(query),
                'result': compress_data(executed_query)
            }
            for db_id, query, executed_query in zip(db_ids, queries, executed_queries)
        ]

        result = SQLiteDBReader._execute_single_query(db_file, insert_sql,
                                                      timeout_s=300,
                                                      allow_write=True,
                                                      params=params)
        if isinstance(result, ExecutorError):
            logger.error(f'Impossible to set cache in the database. error: {result}')

    @staticmethod
    def get_from_cache(db_file: str, db_ids: list[str], queries: list[str]) -> list[OutputTable | NotFoundInCacheError]:
        """Retrieve a file with the given name and parameters."""
        select_sql = """
                     SELECT result
                     FROM `cache_data`
                     WHERE hash_key = ?;
                     """.strip()
        hash_ids = [{'hash_key': hash_db_id_sql(db_id, parse_sql_query(query))}
                    for db_id, query in zip(db_ids, queries)]

        results = SQLiteDBReader.execute_queries(
            db_file,
            queries=[select_sql] * len(hash_ids),
            params=hash_ids,
            timeout_s=100,
        )

        return [uncompress_data(data) for data in results]
