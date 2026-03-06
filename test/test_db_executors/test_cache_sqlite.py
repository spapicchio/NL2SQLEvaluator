import pytest

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import DataToCache, DataToFetch, NotFoundInCacheError
from NL2SQLEvaluator.db_executor_nodes.cache.sqlite_cache import SqliteCache
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SQLExecutorOutput


class TestSqliteCache:
    """Tests for the SqliteCache implementation."""

    @pytest.fixture
    def cache_path(self, tmp_path):
        """Provides a temporary path for the SQLite cache file."""
        return str(tmp_path / "test_cache.db")

    @pytest.fixture
    def cache_instance(self, cache_path):
        """Provides a clean SqliteCache instance."""
        return SqliteCache(cache_path)

    def test_set_and_get_success(self, cache_instance):
        """Verify that data can be saved and retrieved correctly."""
        # Setup mock data
        table = SQLExecutorOutput(columns=["id", "name"], rows=[(1, "Alice"), (2, "Bob")])
        fetch_req = DataToFetch(
            db_path="users_db",
            query="SELECT * FROM users",
            dialect="sqlite"
        )

        data_to_cache = DataToCache[SQLExecutorOutput](
            **fetch_req.model_dump(),
            result=table,
        )

        # Execute
        cache_instance.set_in_cache([data_to_cache])
        results = cache_instance.get_from_cache([fetch_req])

        # Assert
        assert len(results) == 1
        assert isinstance(results[0], SQLExecutorOutput)
        assert results[0].rows == table.rows

    def test_cache_miss(self, cache_instance):
        """Verify behavior when a hash_key is not found."""
        fetch_req = DataToFetch(
            db_path="users_db",
            query="SELECT *",
            dialect="sqlite"
        )
        results = cache_instance.get_from_cache([fetch_req])

        assert len(results) == 1
        assert isinstance(results[0], NotFoundInCacheError)

    def test_batch_operations(self, cache_instance):
        """Verify multiple records can be handled at once."""
        table_a = SQLExecutorOutput(columns=["val"], rows=[("A",)])
        table_b = SQLExecutorOutput(columns=["val"], rows=[("B",)])

        data = [
            DataToCache(db_path="db", query="q1", result=table_a, dialect="sql"),
            DataToCache(db_path="db", query="q2", result=table_b, dialect="sql")
        ]

        cache_instance.set_in_cache(data)

        fetches = [
            DataToFetch(db_path="db", query="q1", dialect="sql"),
            DataToFetch(db_path="db", query="q2", dialect="sql"),
            DataToFetch(db_path="db", query="q3", dialect="sql")
        ]
        results = cache_instance.get_from_cache(fetches)

        assert len(results) == 3
        assert results[0].rows == [("A",)]
        assert results[1].rows == [("B",)]
        assert isinstance(results[2], NotFoundInCacheError)
