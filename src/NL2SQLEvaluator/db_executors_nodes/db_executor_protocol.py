import re
from typing import TypeAlias, Any, Protocol

from NL2SQLEvaluator.sql_cache_nodes.sql_cache_protocol import SQLCacheProtocol

Row: TypeAlias = tuple[Any]
OutputTable: TypeAlias = list[Row]


class ExecutorError(Exception):
    pass


class DbReaderProtocol(Protocol):
    @staticmethod
    def execute_queries(
            db_files: list[str],
            queries: list[list[str]],
            params: list[dict] | None = None,
            cache_db: SQLCacheProtocol | None = None,
            cache_db_file: str | None = None,
            *args, **kwargs
    ) -> list[list[OutputTable | ExecutorError]]:
        ...


def get_last_pattern_or_same(generation: str, pattern: str):
    matches = re.findall(pattern, generation, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    else:
        return generation


def extract_sql_or_same(generation: str):
    sql_from_answer_tag = get_last_pattern_or_same(generation, r"<answer>(.*?)</answer>")
    sql_without_quotes = get_last_pattern_or_same(sql_from_answer_tag, r"```sql(.*?)```")
    sql_cleaned = sql_without_quotes.strip().strip("`").strip()
    return sql_cleaned


def execute_queries_in_model_predictions(
        db_executor: DbReaderProtocol,
        db_files: list[str],
        queries: list[list[str]],
        params: list[dict] | None = None,
        sql_cache_protocol: SQLCacheProtocol | None = None,
        cache_db_file: str | None = None,
        *args, **kwargs
) -> list[list[OutputTable | ExecutorError]]:
    queries = [[extract_sql_or_same(query) for query in query_list] for query_list in queries]
    return db_executor.execute_queries(db_files, queries, params, sql_cache_protocol, cache_db_file, *args, **kwargs)
