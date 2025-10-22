from typing import Protocol, TypedDict


class DataInput(TypedDict):
    db_file: str
    question: str
    target_query: list[str]
    evidence: str | None


class DataReaderProtocol(Protocol):
    @staticmethod
    def read(file_path: str, *args, **kwargs) -> list[DataInput]:
        ...
