"""
This module provides data structures for Code query caching.

It handles normalization of queries across different dialects to ensure
consistent cache key generation.
"""
import hashlib
from typing import Protocol, Generic
from typing import TypeVar

from pydantic import BaseModel, Field, ConfigDict, model_validator

from NL2SQLEvaluator.db_executor_nodes.cache.code_normalizer import (
    QUERY_NORMALIZERS,
    base_normalize_whitespace_and_case
)

T = TypeVar("T")


class NotFoundInCacheError(Exception):
    """Raised when an item is not found in the cache."""
    pass


class DataToFetch(BaseModel):
    """Metadata for identifying and hashing a SQL query in the cache.

    This class normalizes SQL input and generates a unique hash key used
    for cache lookups. It ensures that queries differing only by case or
    whitespace are treated as identical.

    Args:
        db_path: The filesystem path or identifier for the database.
        query: The raw SQL query string.
        dialect: The SQL dialect (e.g., 'sqlite', 'postgres'). Defaults to 'sqlite'.

    Attributes:
        hash_key: A deterministic SHA256 hash of the normalized inputs.

    Example:
        >>> data = DataToFetch(db_path="HR_DB", query="SELECT * FROM table", dialect="sqlite")
        >>> print(data.hash_key)
        'a1b2c3...'
    """

    model_config = ConfigDict(frozen=True)

    db_path: str
    query: str
    dialect: str = "sqlite"
    hash_key: str = Field(default="", init=False, repr=False)

    @model_validator(mode='after')
    def normalize_and_hash(self) -> "DataToFetch":
        """Normalizes data and computes hash after Pydantic type validation.
        Why: By using 'after', we guarantee that inputs are already strings.
        We use object.__setattr__ to modify the fields since the model is frozen.
        Returns:
            The instance with normalized fields and a populated hash_key.
        """
        # Normalize strings
        d_lower = self.dialect.lower()
        p_lower = self.db_path.lower()

        normalizer = QUERY_NORMALIZERS.get(
            d_lower,
            base_normalize_whitespace_and_case
        )
        q_norm = normalizer(self.query, d_lower)

        # Update fields using __setattr__ to bypass frozen=True
        object.__setattr__(self, "dialect", d_lower)
        object.__setattr__(self, "db_path", p_lower)
        object.__setattr__(self, "query", q_norm)

        # Compute and set hash_key
        payload = f"{p_lower}|{d_lower}|{q_norm}".encode("utf-8")
        h = hashlib.sha256(payload).hexdigest()
        object.__setattr__(self, "hash_key", h)
        return self


class DataToCache(DataToFetch, Generic[T]):
    """Container for a query's metadata and its generic execution result.

    Why: Provides a type-safe wrapper for any result type (DataFrames, Tables).
    Example:
        >>> item = DataToCache[int](db_path="db", query="SELECT 1", result=1)
    """
    result: T


class SQLCacheProtocol(Protocol[T]):
    """Interface for cache backends supporting type-safe retrieval.

    Why: Decouples storage logic (SQLite, Redis) from the execution pipeline.
    """

    def set_in_cache(self, cache_path: str, data_to_cache: list[DataToCache[T]]) -> None:
        """Persists a list of results to the cache backend."""
        ...

    def get_from_cache(
            self,
            cache_path: str,
            data_to_fetch: list[DataToFetch]
    ) -> list[DataToCache[T] | NotFoundInCacheError]:
        """Retrieves results, returning DataToCache[T] for hits or Error for misses."""
        ...
