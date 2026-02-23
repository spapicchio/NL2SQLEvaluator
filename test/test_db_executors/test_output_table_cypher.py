import pickle
import zlib

import pytest

from NL2SQLEvaluator.db_executor_nodes.db_executor_output import (
    CypherExecutorOutput,
    SparqlExecutorOutput,
)


class TestCypherOutputTable:
    """Test suite for Cypher-specific result handling and comparisons."""

    def test_equality_unordered(self):
        """Verify that row order is ignored by default (multiset logic)."""
        table_a = CypherExecutorOutput(rows=[("Alice", 1), ("Bob", 2)])
        table_b = CypherExecutorOutput(rows=[("Bob", 2), ("Alice", 1)])
        assert table_a.is_equivalent_to(table_b) is True

    def test_equality_ordered_strict(self):
        """Verify that order matters when is_row_order_important is True."""
        table_a = CypherExecutorOutput(rows=[("Alice", 1), ("Bob", 2)])
        table_b = CypherExecutorOutput(rows=[("Bob", 2), ("Alice", 1)])
        assert table_a.is_equivalent_to(table_b, is_row_order_important=True) is False

    def test_column_permutation(self):
        """Verify that different column orders are matched via permutations."""
        table_a = CypherExecutorOutput(rows=[(1, "Alice"), (2, "Bob")])
        table_b = CypherExecutorOutput(rows=[("Alice", 1), ("Bob", 2)])
        assert table_a.is_equivalent_to(table_b) is True

    def test_complex_types(self):
        """Verify that rows with nested lists/dicts are handled correctly."""
        # Nested lists become tuples during validation
        table_a = CypherExecutorOutput(rows=[([1, 2, 3], "x")])
        table_b = CypherExecutorOutput(rows=[([1, 2, 3], "x")])
        assert table_a.is_equivalent_to(table_b) is True

    def test_complex_types_dict(self):
        """Verify that rows with nested dicts are normalized."""
        table_a = CypherExecutorOutput(rows=[({"a": 1, "b": 2},)])
        table_b = CypherExecutorOutput(rows=[({"b": 2, "a": 1},)])
        assert table_a.is_equivalent_to(table_b) is True

    def test_multiset_duplication(self):
        """Verify that row counts must match (not just a set comparison)."""
        table_a = CypherExecutorOutput(rows=[("Alice", 1), ("Alice", 1)])
        table_b = CypherExecutorOutput(rows=[("Alice", 1)])
        assert table_a.is_equivalent_to(table_b) is False

    def test_serialization_roundtrip(self):
        """Ensure data integrity through compression and decompression."""
        original = CypherExecutorOutput(rows=[(1, "Alice"), (2, "Bob")])
        compressed = original.compress()
        decompressed = CypherExecutorOutput.decompress(compressed)

        assert decompressed.rows == original.rows

    def test_container_methods(self):
        """Test the dunder methods for list-like access."""
        table = CypherExecutorOutput(rows=[(1,), (2,), (3,)])
        assert len(table) == 3
        assert table[0] == (1,)
        assert (2,) in table
        assert table(slice(0, 2)) == [(1,), (2,)]

    def test_empty_results_equal(self):
        """Two empty results should be equivalent."""
        table_a = CypherExecutorOutput(rows=[])
        table_b = CypherExecutorOutput(rows=[])
        assert table_a.is_equivalent_to(table_b) is True

    def test_different_types_not_comparable(self):
        """CypherExecutorOutput should not be equivalent to SparqlExecutorOutput."""
        table_a = CypherExecutorOutput(rows=[(1,)])
        table_b = SparqlExecutorOutput(rows=[(1,)])
        assert table_a.is_equivalent_to(table_b) is False

    def test_different_column_counts(self):
        """Different number of columns should not match."""
        table_a = CypherExecutorOutput(rows=[(1, 2)])
        table_b = CypherExecutorOutput(rows=[(1, 2, 3)])
        assert table_a.is_equivalent_to(table_b) is False

    def test_decompress_invalid_bytes(self):
        """Ensures that passing garbage data to decompress fails gracefully."""
        with pytest.raises((zlib.error, pickle.UnpicklingError, EOFError)):  # pyrefly: ignore
            CypherExecutorOutput.decompress(b"invalid_compressed_stream_12345")


class TestCypherOutputTableRobustness:
    """Edge-case testing for CypherExecutorOutput serialization."""

    def test_compress_empty_table(self):
        """Checks if the system handles empty rows."""
        table = CypherExecutorOutput(rows=[])
        compressed = table.compress()
        decompressed = CypherExecutorOutput.decompress(compressed)
        assert decompressed.rows == []

    def test_decompress_truncated_data(self):
        """Checks behavior when the byte stream is cut short."""
        table = CypherExecutorOutput(rows=[(1,)] * 100)
        compressed = table.compress()
        truncated = compressed[:len(compressed) // 2]
        with pytest.raises((zlib.error, pickle.UnpicklingError)):  # pyrefly: ignore
            CypherExecutorOutput.decompress(truncated)
