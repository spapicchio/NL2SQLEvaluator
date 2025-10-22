from typing import TypeAlias, Any, Protocol

Row: TypeAlias = tuple[Any]
OutputTable: TypeAlias = list[Row]


class DbReaderProtocol(Protocol):
    def execute_queries(self, db_file: str, queries: list[str], *args, **kwargs) -> list[OutputTable]:
        ...
