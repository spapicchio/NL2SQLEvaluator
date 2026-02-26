import pickle
import random
import string
import zlib

import pytest

from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SQLExecutorOutput


class TestSQLOutputTable:
    """Test suite for SQL-specific result handling and comparisons."""

    def test_equality_unordered(self):
        """Verify that row order is ignored by default (multiset logic)."""
        table_a = SQLExecutorOutput(rows=[(1, "A", None), (2, "B")])
        table_b = SQLExecutorOutput(rows=[(2, "B"), (1, "A", None)])
        assert table_a.is_equivalent_to(table_b) is True

    def test_equality_ordered_strict(self):
        """Verify that order matters when is_row_order_important is True."""
        table_a = SQLExecutorOutput(rows=[(1, "A"), (2, "B")])
        table_b = SQLExecutorOutput(rows=[(2, "B"), (1, "A")])
        assert table_a.is_equivalent_to(table_b, is_row_order_important=True) is False

    def test_equality_diff_projected_col(self):
        """Verify that order matters when is_row_order_important is True."""
        table_a = SQLExecutorOutput(rows=[(None, 1, "A"), ("B", 2)])
        table_b = SQLExecutorOutput(rows=[(2, "B"), ("A", None, 1)])
        assert table_a.is_equivalent_to(table_b, is_row_order_important=False) is True

    def test_multiset_duplication(self):
        """Verify that row counts must match (not just a set comparison)."""
        table_a = SQLExecutorOutput(rows=[(1, "A"), (1, "A")])
        table_b = SQLExecutorOutput(rows=[(1, "A")])
        assert table_a.is_equivalent_to(table_b) is False

    def test_serialization_roundtrip(self):
        """Ensure data integrity through compression and decompression."""
        original = SQLExecutorOutput(rows=[(1, "Alice"), (2, "Bob")], columns=["id", "name"])
        compressed = original.compress()
        decompressed = SQLExecutorOutput.decompress(compressed)

        assert decompressed.rows == original.rows

    def test_invalid_types_validation(self):
        """Ensure Pydantic catches non-hashable types in rows."""
        with pytest.raises(TypeError, match="hashable types"):
            SQLExecutorOutput(rows=[(1, ["inner", "list"])])

    def test_container_methods(self):
        """Test the dunder methods for list-like access."""
        table = SQLExecutorOutput(rows=[(1,), (2,), (3,)])
        assert len(table) == 3
        assert table[0] == (1,)
        assert (2,) in table
        assert table(slice(0, 2)) == [(1,), (2,)]


class TestSQLOutputTableRobustness:
    """Edge-case testing for SQLOutputTable serialization and capacity."""

    def test_compress_empty_table(self):
        """Checks if the system handles empty lists for columns and rows."""
        table = SQLExecutorOutput(columns=[], rows=[])
        compressed = table.compress()

        decompressed = table.decompress(compressed)
        assert decompressed.rows == []

    def test_compress_large_result_set(self):
        """Stress test with 100,000 rows to check performance and stability.

        Why: Ensures zlib doesn't hit a buffer limit and pickle handles scale.
        """
        # Create 100k rows: (int, string, float)
        large_rows = [
            (i, "".join(random.choices(string.ascii_letters, k=10)), random.random())
            for i in range(100_000)
        ]
        table = SQLExecutorOutput(columns=["id", "token", "score"], rows=large_rows)

        compressed = table.compress()

        # Ensure compression actually reduced the footprint compared to raw pickle
        raw_pickle_size = len(pickle.dumps(table.model_dump()))
        assert len(compressed) < raw_pickle_size

        decompressed = table.decompress(compressed)
        assert len(decompressed.rows) == 100_000
        assert decompressed.rows[500] == large_rows[500]

    def test_decompress_invalid_bytes(self):
        """Ensures that passing garbage data to decompress fails gracefully."""
        table = SQLExecutorOutput(columns=["a"], rows=[(1,)])

        with pytest.raises((zlib.error, pickle.UnpicklingError, EOFError)):  # pyrefly: ignore
            # Passing random non-compressed bytes
            table.decompress(b"invalid_compressed_stream_12345")

    def test_decompress_truncated_data(self):
        """Checks behavior when the byte stream is cut short."""
        table = SQLExecutorOutput(columns=["a"], rows=[(1,)] * 100)
        compressed = table.compress()

        # Truncate the byte string
        truncated = compressed[:len(compressed) // 2]
        with pytest.raises((zlib.error, pickle.UnpicklingError)):  # pyrefly: ignore
            table.decompress(truncated)
