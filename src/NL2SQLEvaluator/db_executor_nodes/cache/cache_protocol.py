"""Module for managing cache-related data structures and protocols."""

import hashlib
from typing import Self, Protocol

from pydantic import BaseModel, model_validator

from NL2SQLEvaluator.db_executor_nodes.cache.code_normalizer import (
    QUERY_NORMALIZERS,
    base_normalize_whitespace_and_case
)
from NL2SQLEvaluator.db_executor_nodes.output_table import GenericOutputTable


class NotFoundInCacheError(Exception):
    """Raised when an item is not found in the cache."""
    pass


class DataToFetch(BaseModel):
    """
    Metadata for identifying a query in the cache.

    Why: Ensures that queries differing only in whitespace or case are 
    treated as hits by generating a deterministic hash key.

    Example:
        >>> fetch = DataToFetch(db_path="HR_DB", query="SELECT * FROM users", dialect="sqlite")
        >>> print(fetch.hash_key)
    """
    db_path: str
    query: str
    dialect: str = "sqlite"

    @model_validator(mode='after')
    def normalize_query(self) -> Self:
        """Normalizes the query string based on the SQL dialect."""
        normalizer = QUERY_NORMALIZERS.get(
            self.dialect.lower(),
            base_normalize_whitespace_and_case
        )
        self.query = normalizer(self.query, self.dialect)
        return self

    @property
    def hash_key(self) -> str:
        """Returns a SHA256 hash of the unique query identifiers."""
        payload = f"{self.db_path}|{self.dialect.lower()}|{self.query}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class DataToCache(DataToFetch):
    """Container for storing both the query metadata and its executed result."""
    result: GenericOutputTable


class SQLCacheProtocol(Protocol):
    """
    Defines the interface for cache storage backends.
    """

    def set_in_cache(self, cache_path: str, data_to_cache: list[DataToCache]) -> None:
        """
        Persists a list of query results to the specified cache.

        Args:
            cache_path: The destination identifier (e.g., directory path or URI).
            data_to_cache: The data objects to be stored.
        """
        pass

    def get_from_cache(
            self,
            cache_path: str,
            data_to_fetch: list[DataToFetch]
    ) -> list[DataToCache | NotFoundInCacheError]:
        """
        Retrieves cached results for the requested queries.

        Args:
            cache_path: The source identifier for the cache.
            data_to_fetch: Metadata for the queries to look up.

        Returns:
            A list where each element is either the cached result or an error object.
        """
        pass
