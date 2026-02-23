import pickle
import zlib

import pytest

from NL2SQLEvaluator.db_executor_nodes.db_executor_output import (
    CypherExecutorOutput,
    SparqlExecutorOutput,
)


class TestSparqlOutputTable:
    """Test suite for SPARQL-specific result handling and comparisons."""

    def test_equality_unordered(self):
        """Verify that row order is ignored by default."""
        table_a = SparqlExecutorOutput(rows=[("http://ex.org/1", "Alice"), ("http://ex.org/2", "Bob")])
        table_b = SparqlExecutorOutput(rows=[("http://ex.org/2", "Bob"), ("http://ex.org/1", "Alice")])
        assert table_a.is_equivalent_to(table_b) is True

    def test_equality_ordered_strict(self):
        """Verify that order matters when is_row_order_important is True."""
        table_a = SparqlExecutorOutput(rows=[("http://ex.org/1", "Alice"), ("http://ex.org/2", "Bob")])
        table_b = SparqlExecutorOutput(rows=[("http://ex.org/2", "Bob"), ("http://ex.org/1", "Alice")])
        assert table_a.is_equivalent_to(table_b, is_row_order_important=True) is False

    def test_column_permutation(self):
        """Verify that different column orders are matched."""
        table_a = SparqlExecutorOutput(rows=[(1, "Alice"), (2, "Bob")])
        table_b = SparqlExecutorOutput(rows=[("Alice", 1), ("Bob", 2)])
        assert table_a.is_equivalent_to(table_b) is True

    def test_multiset_duplication(self):
        """Verify that row counts must match."""
        table_a = SparqlExecutorOutput(rows=[("Alice",), ("Alice",)])
        table_b = SparqlExecutorOutput(rows=[("Alice",)])
        assert table_a.is_equivalent_to(table_b) is False

    def test_serialization_roundtrip(self):
        """Ensure data integrity through compression and decompression."""
        original = SparqlExecutorOutput(rows=[("http://ex.org/1", "Alice")])
        compressed = original.compress()
        decompressed = SparqlExecutorOutput.decompress(compressed)

        assert decompressed.rows == original.rows

    def test_container_methods(self):
        """Test the dunder methods for list-like access."""
        table = SparqlExecutorOutput(rows=[("a",), ("b",), ("c",)])
        assert len(table) == 3
        assert table[0] == ("a",)
        assert ("b",) in table
        assert table(slice(0, 2)) == [("a",), ("b",)]

    def test_empty_results_equal(self):
        """Two empty results should be equivalent."""
        table_a = SparqlExecutorOutput(rows=[])
        table_b = SparqlExecutorOutput(rows=[])
        assert table_a.is_equivalent_to(table_b) is True

    def test_different_types_not_comparable(self):
        """SparqlExecutorOutput should not be equivalent to CypherExecutorOutput."""
        table_a = SparqlExecutorOutput(rows=[(1,)])
        table_b = CypherExecutorOutput(rows=[(1,)])
        assert table_a.is_equivalent_to(table_b) is False

    def test_different_column_counts(self):
        """Different number of columns should not match."""
        table_a = SparqlExecutorOutput(rows=[(1, 2)])
        table_b = SparqlExecutorOutput(rows=[(1, 2, 3)])
        assert table_a.is_equivalent_to(table_b) is False

    def test_decompress_invalid_bytes(self):
        """Ensures that passing garbage data to decompress fails gracefully."""
        with pytest.raises((zlib.error, pickle.UnpicklingError, EOFError)):  # pyrefly: ignore
            SparqlExecutorOutput.decompress(b"invalid_compressed_stream_12345")


class TestSparqlOutputTableRobustness:
    """Edge-case testing for SparqlExecutorOutput serialization."""

    def test_compress_empty_table(self):
        """Checks if the system handles empty rows."""
        table = SparqlExecutorOutput(rows=[])
        compressed = table.compress()
        decompressed = SparqlExecutorOutput.decompress(compressed)
        assert decompressed.rows == []

    def test_decompress_truncated_data(self):
        """Checks behavior when the byte stream is cut short."""
        table = SparqlExecutorOutput(rows=[("uri",)] * 100)
        compressed = table.compress()
        truncated = compressed[:len(compressed) // 2]
        with pytest.raises((zlib.error, pickle.UnpicklingError)):  # pyrefly: ignore
            SparqlExecutorOutput.decompress(truncated)
