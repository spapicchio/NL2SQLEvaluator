import re
from typing import Protocol, Any, Iterator, Self

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import OutputTable, SQLCacheProtocol
from pydantic import BaseModel, model_validator


class ExecutorError(Exception):
    pass


class ExecuteTask(BaseModel):
    db_files: list[str] | str
    queries: list[str]
    params: list[dict] | dict | None = None
    db_ids: list[str] | None = None
    timeout: float | int | list[float | int] = 500

    # validate target_sql, inputs seq and db_files all have same length
    @model_validator(mode='after')
    def check_len(self) -> Self:
        if isinstance(self.db_files, str):
            self.db_files = [self.db_files for _ in self.queries]

        if self.params is None:
            self.params = [{} for _ in self.queries]
        if isinstance(self.timeout, (float, int)):
            self.timeout = [self.timeout for _ in self.queries]
        elif isinstance(self.params, dict):
            self.params = [self.params for _ in self.queries]

        len_queries = len(self.queries)
        if not len(self.db_files) == len_queries:
            raise ValueError(f"Length of db_files {len(self.db_files)} must match length of queries {len_queries}.")
        if not len(self.timeout) == len_queries:
            raise ValueError("Length of timeout must match length of queries.")
        if not len(self.params) == len_queries:
            raise ValueError("Length of params must match length of queries.")
        if self.db_ids is not None and len(self.db_ids) == len_queries:
            raise ValueError("Length of db_ids must match length of queries.")

        return self

    def __iter__(self) -> Iterator[Any]:
        return iter(zip(self.queries, self.params, self.db_files, self.timeout))


class DbReaderProtocol(Protocol):
    @staticmethod
    def execute_queries(
            tasks: list[ExecuteTask],
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
    assert len(db_files) == len(queries)
    tasks = [
        ExecuteTask(
            db_files=db_files[i],
            queries=[extract_sql_or_same(query) for query in queries[i]],
            params=params[i] if params is not None else None
        )
        for i in range(len(db_files))
    ]

    return db_executor.execute_queries(tasks=tasks, cache_db=sql_cache_protocol, cache_db_file=cache_db_file,
                                       *args, **kwargs)
