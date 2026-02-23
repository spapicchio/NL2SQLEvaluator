"""Module for standardized database execution across multiple languages.

This module defines an abstract interface for executors to ensure that SQL,
Cypher, and SPARQL queries always return a uniform GenericOutputTable.
"""

import pickle
import zlib
from abc import ABC
from collections import Counter, defaultdict
from typing import Self, Any, Optional, override

from pydantic import BaseModel, model_validator, ConfigDict

from NL2SQLEvaluator.logger import get_logger

logger = get_logger(__name__)


class ExecutorError(Exception):
    """Raised when a database execution fails or times out."""
    pass


class GenericExecutorOutput(BaseModel, ABC):
    """Abstract base class representing a standardized database result set.

    Provides shared logic for caching (compress/decompress), comparison
    (multiset with column-permutation support), and container access.

    Subclasses must implement a model_validator to normalize language-specific
    types (e.g., Neo4j Date → str, rdflib URIRef → str).

    Attributes:
        rows: A list of tuples containing the result data.
        execution_time: Optional float representing query duration.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rows: list[tuple[Any, ...]]
    execution_time: Optional[float] = None

    # ── Comparison ──────────────────────────────────────────────────

    def is_equivalent_to(self, other: 'GenericExecutorOutput', *args, **kwargs) -> bool:
        """Compare two result sets as multisets with column-permutation support.

        Uses signature-constrained backtracking instead of brute-force O(n!)
        permutations: columns are grouped by their value distribution, and only
        columns with identical distributions are permuted against each other.

        Subclasses may override for language-specific semantics (e.g.,
        SQLExecutorOutput uses row-value sorting without column permutation).

        Args:
            other: The other result to compare against.
            **kwargs:
                is_row_order_important (bool): If True, row order matters. Defaults to False.
        """
        is_row_order_important = kwargs.get('is_row_order_important', False)

        if not isinstance(other, type(self)):
            logger.warning('Comparison between different output table types. Returning False.')
            return False

        if len(self.rows) != len(other.rows):
            return False

        if not self.rows:
            return True

        num_cols = len(self.rows[0])
        if num_cols != len(other.rows[0]):
            return False

        for perm in self._valid_column_mappings(self.rows, other.rows, num_cols):
            permuted_other = [tuple(row[i] for i in perm) for row in other.rows]

            self_h = [self.to_hashable(row) for row in self.rows]
            other_h = [self.to_hashable(row) for row in permuted_other]

            if is_row_order_important:
                if self_h == other_h:
                    return True
            else:
                if Counter(self_h) == Counter(other_h):
                    return True

        return False

    # ── Serialization ───────────────────────────────────────────────

    def compress(self, *args, **kwargs) -> bytes:
        """Compress the result using zlib + pickle for cache storage."""
        return zlib.compress(pickle.dumps(self.model_dump()))

    @classmethod
    def decompress(cls, compressed_data: bytes, *args, **kwargs) -> Self:
        """Reconstruct an instance from compressed bytes."""
        state = pickle.loads(zlib.decompress(compressed_data))
        return cls(**state)

    # ── Utility ─────────────────────────────────────────────────────

    @staticmethod
    def to_hashable(val: Any) -> Any:
        """Recursively convert a value to a hashable representation."""
        if isinstance(val, dict):
            return tuple(sorted(
                ((k, GenericExecutorOutput.to_hashable(v)) for k, v in val.items()),
                key=lambda x: str(x[0])
            ))
        if isinstance(val, (list, tuple, set)):
            return tuple(GenericExecutorOutput.to_hashable(item) for item in val)
        return val

    @staticmethod
    def unorder_row(row: tuple) -> tuple:
        """Sort values within a row for permutation-invariant canonicalization.

        Used by majority voting to canonicalize rows independently of column order.
        NOT used by is_equivalent_to (which does proper column permutation instead).
        """
        return tuple(sorted(row, key=lambda x: str(GenericExecutorOutput.to_hashable(x))))

    @staticmethod
    def _valid_column_mappings(self_rows, other_rows, num_cols):
        """Yield column permutations constrained by column-value signatures.

        Instead of brute-forcing all n! permutations, groups columns by their
        value distribution (multiset of stringified values) and only permutes
        within groups of identical signatures. This reduces O(n!) to
        O(k1! * k2! * ...) where ki are group sizes.
        """
        if num_cols == 0:
            yield ()
            return

        def _col_sig(rows, idx):
            vals = [GenericExecutorOutput.to_hashable(row[idx]) for row in rows]
            return tuple(sorted(vals, key=str))

        self_sigs = [_col_sig(self_rows, i) for i in range(num_cols)]
        other_sigs = [_col_sig(other_rows, i) for i in range(num_cols)]

        # Map signature → list of other-column indices
        sig_to_other = defaultdict(list)
        for i, sig in enumerate(other_sigs):
            sig_to_other[sig].append(i)

        # For each self column, find candidate other columns with matching signature
        candidates = []
        for sig in self_sigs:
            c = sig_to_other.get(sig)
            if not c:
                return  # No valid mapping exists
            candidates.append(c)

        # Backtrack: assign one other-column per self-column (1-to-1)
        def _bt(depth, used, current):
            if depth == num_cols:
                yield tuple(current)
                return
            for col in candidates[depth]:
                if col not in used:
                    current.append(col)
                    used.add(col)
                    yield from _bt(depth + 1, used, current)
                    current.pop()
                    used.discard(col)

        yield from _bt(0, set(), [])

    # ── Container protocol ──────────────────────────────────────────

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
    Uses row-value sorting (not column permutation) since SQL column order
    is determined by the SELECT clause.

    Example:
        >>> table = SQLExecutorOutput(rows=[(1,), (2,)])
        >>> table.is_equivalent_to(SQLExecutorOutput(rows=[(2,), (1,)]))
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
        """Compare SQL results as multisets with within-row value sorting.

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


class CypherExecutorOutput(GenericExecutorOutput):
    """Wraps Neo4j Cypher query results for deterministic comparison and caching.

    Neo4j returns records as list[dict]. During construction, each dict is sorted
    by key alphabetically and values are extracted as a tuple. Complex Neo4j types
    (Date, DateTime, nested lists/dicts) are recursively converted to hashable primitives.

    Inherits column-permutation comparison from GenericExecutorOutput.
    """

    @model_validator(mode='after')
    def normalize_neo4j_types(self) -> Self:
        """Recursively converts Neo4j-specific types to hashable primitives."""
        self.rows = [tuple(self._to_primitive(v) for v in row) for row in self.rows]
        return self

    @staticmethod
    def _to_primitive(val: Any) -> Any:
        """Recursively convert Neo4j/complex types to hashable primitives."""
        try:
            import neo4j.time as neo4j_time
            if isinstance(val, (neo4j_time.Date, neo4j_time.DateTime)):
                return val.iso_format()
        except ImportError:
            pass

        if isinstance(val, dict):
            return tuple(sorted(
                ((k, CypherExecutorOutput._to_primitive(v)) for k, v in val.items()),
                key=lambda x: str(x[0])
            ))
        if isinstance(val, (list, set)):
            return tuple(CypherExecutorOutput._to_primitive(item) for item in val)
        return val


class SparqlExecutorOutput(GenericExecutorOutput):
    """Wraps SPARQL SELECT query results for deterministic comparison and caching.

    SPARQL SELECT returns bindings as list[dict]. During construction, each binding
    dict is sorted by key and values are extracted as a tuple. RDF types (URIs,
    typed literals) are converted to native Python primitives.

    Inherits column-permutation comparison from GenericExecutorOutput.
    """

    @model_validator(mode='after')
    def normalize_rdf_types(self) -> Self:
        """Convert RDF types to hashable primitives."""
        self.rows = [tuple(self._to_primitive(v) for v in row) for row in self.rows]
        return self

    @staticmethod
    def _to_primitive(val: Any) -> Any:
        """Convert RDF/complex types to hashable primitives."""
        try:
            from rdflib import URIRef, Literal, BNode
            if isinstance(val, (URIRef, BNode)):
                return str(val)
            if isinstance(val, Literal):
                try:
                    return val.toPython()
                except Exception:
                    return str(val)
        except ImportError:
            pass

        if isinstance(val, dict):
            # Handle SPARQLWrapper JSON result binding format
            if 'value' in val and 'type' in val:
                return val['value']
            return tuple(sorted(
                ((k, SparqlExecutorOutput._to_primitive(v)) for k, v in val.items()),
                key=lambda x: str(x[0])
            ))
        if isinstance(val, (list, set)):
            return tuple(SparqlExecutorOutput._to_primitive(item) for item in val)
        return val
