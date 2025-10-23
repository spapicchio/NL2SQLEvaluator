from typing import TypeAlias, Any, Protocol

Row: TypeAlias = tuple[Any]
OutputTable: TypeAlias = list[Row]


class ExecutorError(Exception):
    pass


class DbReaderProtocol(Protocol):
    @staticmethod
    def execute_queries(db_file: str, queries: list[str], params: list[dict] | None = None, *args, **kwargs) -> list[
        OutputTable | ExecutorError]:
        ...
