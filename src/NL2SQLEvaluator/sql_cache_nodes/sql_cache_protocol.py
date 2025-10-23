from typing import Protocol

from NL2SQLEvaluator.db_executors_nodes.db_executor_protocol import OutputTable


class NotFoundInCacheError(Exception):
    """Raised when an item is not found in the cache."""
    pass


class SQLCacheProtocol(Protocol):
    @staticmethod
    def set_in_cache(db_file: str, db_ids: list[str], queries: list[str],
                     executed_query: list[OutputTable]) -> None:
        """Save a file with the given name and parameters."""
        ...

    @staticmethod
    def get_from_cache(db_file: str, db_ids: list[str], queries: list[str]) -> list[OutputTable | NotFoundInCacheError]:
        """Retrieve a file with the given name and parameters."""
        ...
