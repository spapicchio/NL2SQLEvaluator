"""Module for standardized database execution across multiple languages.

This module defines an abstract interface for executors to ensure that SQL,
Cypher, and SPARQL queries always return a uniform GenericOutputTable.
"""

from abc import ABC, abstractmethod
from collections import Counter
from typing import Self, Any, Optional, override

from pydantic import BaseModel, model_validator, ConfigDict

from NL2SQLEvaluator.logger import get_logger

logger = get_logger(__name__)


class ExecutorError(Exception):
    """Raised when a database execution fails or times out."""
    pass


class GenericExecutorOutput(BaseModel, ABC):
    """Abstract base class representing a standardized database result set.

    This class provides a consistent interface for caching and evaluating results
    regardless of the underlying database engine (SQL, Neo4j, etc.).

    Attributes:
        columns: A list of column headers.
        rows: A list of tuples containing the result data.
        execution_time: Optional float representing query duration.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rows: list[tuple[Any, ...]]
    execution_time: Optional[float] = None

    @abstractmethod
    def is_equivalent_to(self, other_out_executed_code: 'GenericExecutorOutput', *args, **kwargs) -> bool:
        """Compare two result sets based on language-specific rules."""
        pass

    @abstractmethod
    def compress(self, *args, **kwargs) -> bytes:
        """Compress the result to save space in cache."""
        pass

    @staticmethod
    @abstractmethod
    def decompress(compressed_data: bytes, *args, **kwargs) -> 'GenericExecutorOutput':
        """Decompress the result from the internal compressed_data buffer."""
        pass

    def __len__(self):
        return len(self.rows)

    def __call__(self, index: int | slice | None = None) -> Any:
        """Access rows by index or slice. Returns all rows if index is None."""
        if index is None:
            return self.rows
        if isinstance(index, (int, slice)):
            return self.rows[index]
        raise TypeError("index must be an int, slice, or None")

    def __getitem__(self, item):
        return self.rows[item]

    def __contains__(self, item: Any) -> bool:
        return item in self.rows


class SQLExecutorOutput(GenericExecutorOutput):
    """Wraps SQL query results to enable deterministic comparisons and storage.

    SQL results are often unordered by default. This class allows comparing
    two result sets for equality by treating them as multisets.

    Example:
        >>> table = SQLExecutorOutput(columns=["id"], rows=[(1,), (2,)])
        >>> table.is_equivalent_to(SQLExecutorOutput(columns=["id"], rows=[(2,), (1,)]))
        True
    """

    @model_validator(mode='after')
    def forbid_inner_lists(self) -> Self:
        """Ensures that row values are primitive and hashable for canonicalization.

        Raises:
            TypeError: If a row contains a list, set, or dict.

        Returns:
            Self: The validated instance.
        """
        for row in self.rows:
            if any(isinstance(val, (list, set, dict)) for val in row):
                raise TypeError("SQL rows must contain only flat, hashable types.")
        return self

    @override
    def is_equivalent_to(self, other_out_executed_code: 'GenericExecutorOutput', *args, **kwargs) -> bool:
        """Wraps SQL query results to enable deterministic comparisons and storage.

            SQL results are often unordered unless an ORDER BY clause is present. This
            class allows comparing two result sets for equality by treating them as
            multisets (bags), ensuring that row order doesn't break tests unless specified.

            Example:
                >>> table_a = SQLExecutorOutput(rows=[(1, 'Alice'), (2, 'Bob')])
                >>> table_b = SQLExecutorOutput(rows=[(2, 'Bob'), (1, 'Alice')])
                >>> table_a.is_equivalent_to(table_b, is_row_order_important=False)
                True

            Note:
                is_equivalente is also ordering the attribute columns.
                Therefore, it must be used only to compare the execution of SQL code rather than two tables in general.
            """
        if 'is_row_order_important' not in kwargs:
            logger.debug(
                "Missing required argument: `is_row_order_important`"
                " used to determine if row order matters for comparison. Default to FALSE."
            )

        is_row_order_important = kwargs.get('is_row_order_important', False)

        if not isinstance(other_out_executed_code, SQLExecutorOutput):
            logger.warning('Comparison between different output table types is not supported. Returning False.')
            return False

        if len(self.rows) != len(other_out_executed_code.rows):
            return False
        sorted_rows = list(map(self.sort_with_universal_key, self.rows))
        sorted_other_rows = list(map(self.sort_with_universal_key, other_out_executed_code.rows))

        if is_row_order_important:
            return sorted_rows == sorted_other_rows

        return Counter(sorted_rows) == Counter(sorted_other_rows)

    @override
    def compress(self, *args, **kwargs) -> bytes:
        """Compresses the object state using zlib and pickle.

        Returns:
            bytes: The compressed byte string.
        """
        import zlib
        import pickle
        state = self.model_dump()
        return zlib.compress(pickle.dumps(state))

    @staticmethod
    @override
    def decompress(compressed_data: bytes, *args, **kwargs) -> 'SQLExecutorOutput':
        """Reconstructs the instance from a compressed byte string.

        Args:
            compressed_data: The bytes produced by `compress`.
            **kwargs: For interface compatibility.

        Returns:
            Self: A new instance populated with the decompressed data.
        """
        import zlib
        import pickle
        state = pickle.loads(zlib.decompress(compressed_data))
        return SQLExecutorOutput(**state)

    @staticmethod
    def universal_sort_key(x: Any) -> tuple[int, Any]:
        """Provides a sort key to handle mixed types (None, int, str).

        Args:
            x: The value to generate a key for.

        Returns:
            tuple: (priority_int, stringified_or_actual_value).
        """
        if x is None:
            return 0, ''
        elif isinstance(x, (int, float)):
            return 1, float(x)
        else:
            return 2, str(x)

    @staticmethod
    def sort_with_universal_key(arr: tuple) -> tuple:
        """Sorts an individual row tuple using universal_sort_key.

        Args:
            arr: The row tuple to sort.

        Returns:
            tuple: The sorted row.
        """
        return tuple(sorted(arr, key=SQLExecutorOutput.universal_sort_key))
