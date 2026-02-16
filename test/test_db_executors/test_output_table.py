import pickle
import random
import string
import zlib

import pytest

from NL2SQLEvaluator.db_executor_nodes.output_table import SQLOutputTable


class TestSQLOutputTable:
    """Suite for testing SQL-specific result table logic and normalization."""

    @pytest.fixture
    def sample_data(self):
        return {
            "columns": ["id", "name"],
            "rows": [(1, "Alice"), (2, "Bob")]
        }

    def test_initialization_and_access(self, sample_data):
        """Tests basic instantiation and the __call__/__getitem__ overrides."""
        table = SQLOutputTable(**sample_data)
        assert len(table) == 2
        assert table[0] == (1, "Alice")
        assert table() == sample_data["rows"]
        assert table(0) == (1, "Alice")
        assert list(table) == sample_data["rows"]

    def test_forbid_inner_lists(self):
        """Ensures Pydantic validator catches non-hashable/nested types."""
        with pytest.raises(TypeError, match="SQL rows must contain only flat, hashable types."):
            SQLOutputTable(columns=["id"], rows=[(1, [1, 2, 3])])

    @pytest.mark.parametrize("rows_a, rows_b, order_matters, expected", [
        # Same data, same order
        ([(1, "A"), (2, "B")], [(1, "A"), (2, "B")], True, True),
        # Same data, different row order
        ([(1, "A"), (2, "B")], [(2, "B"), (1, "A")], False, True),
        # Same data, different row order (order matters)
        ([(1, "A"), (2, "B")], [(2, "B"), (1, "A")], True, False),
        # Different column order within rows
        ([(1, "A")], [("A", 1)], False, True),
        ([(1, None, 'b')], [(None, 1, 'b')], False, True),
        (
                [
                    (1, None, 'b'),
                    (1, None, 'b'),
                    (1, 2, 'b'),
                    (1, 2, 'b'),
                ],
                [
                    (1, 'b', 2),
                    (1, None, 'b'),
                    (1, 'b', 2),
                    (1, None, 'b'),
                ],
                False,
                True
        ),
        (
                [
                    (1, None, 'b'),
                    (1, None, 'b'),
                    (1, 2, 'b'),
                    (1, 2, 'b'),
                ],
                [
                    (1, 'b', 2),
                    (1, None, 'b'),
                    (1, 'b', 2),
                    (1, None, 'b'),
                ],
                True,
                False
        ),
    ])
    def test_equivalence(self, rows_a, rows_b, order_matters, expected):
        """Tests that table comparison handles ordering and types correctly."""
        t1 = SQLOutputTable(columns=["col1", "col2"], rows=rows_a)
        t2 = SQLOutputTable(columns=["col1", "col2"], rows=rows_b)
        assert t1.is_equivalent_to(t2, is_row_order_important=order_matters) == expected

    def test_compression_roundtrip(self, sample_data):
        """Tests that data survives the zlib/pickle compression cycle."""
        t1 = SQLOutputTable(**sample_data)
        compressed = t1.compress()
        assert isinstance(compressed, bytes)

        t2 = t1.decompress(compressed)
        assert t2.columns == t1.columns
        assert t2.rows == t1.rows
        assert t1.is_equivalent_to(t2)

    def test_universal_sort_key(self):
        """Tests sorting logic for mixed types and Nones."""
        items = [10.5, None, "apple", 1, "banana"]
        # Expected order based on priority: None (0), Numbers (1), Strings (2)
        sorted_items = sorted(items, key=SQLOutputTable.universal_sort_key)
        assert sorted_items[0] is None
        assert sorted_items[1] == 1
        assert sorted_items[-1] == "banana"


class TestSQLOutputTableRobustness:
    """Edge-case testing for SQLOutputTable serialization and capacity."""

    def test_compress_empty_table(self):
        """Checks if the system handles empty lists for columns and rows."""
        table = SQLOutputTable(columns=[], rows=[])
        compressed = table.compress()

        decompressed = table.decompress(compressed)
        assert decompressed.rows == []
        assert decompressed.columns == []

    def test_compress_large_result_set(self):
        """Stress test with 100,000 rows to check performance and stability.

        Why: Ensures zlib doesn't hit a buffer limit and pickle handles scale.
        """
        # Create 100k rows: (int, string, float)
        large_rows = [
            (i, "".join(random.choices(string.ascii_letters, k=10)), random.random())
            for i in range(100_000)
        ]
        table = SQLOutputTable(columns=["id", "token", "score"], rows=large_rows)

        compressed = table.compress()
        # Ensure compression actually reduced the footprint compared to raw pickle
        raw_pickle_size = len(pickle.dumps(table.model_dump()))
        assert len(compressed) < raw_pickle_size

        decompressed = table.decompress(compressed)
        assert len(decompressed.rows) == 100_000
        assert decompressed.rows[500] == large_rows[500]

    def test_decompress_invalid_bytes(self):
        """Ensures that passing garbage data to decompress fails gracefully."""
        table = SQLOutputTable(columns=["a"], rows=[(1,)])

        with pytest.raises((zlib.error, pickle.UnpicklingError, EOFError)):
            # Passing random non-compressed bytes
            table.decompress(b"invalid_compressed_stream_12345")

    def test_decompress_truncated_data(self):
        """Checks behavior when the byte stream is cut short."""
        table = SQLOutputTable(columns=["a"], rows=[(1,)] * 100)
        compressed = table.compress()

        # Truncate the byte string
        truncated = compressed[:len(compressed) // 2]
        with pytest.raises((zlib.error, pickle.UnpicklingError)):
            table.decompress(truncated)
