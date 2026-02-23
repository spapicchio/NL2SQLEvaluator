import pytest

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import DataToCache, DataToFetch, NotFoundInCacheError
from NL2SQLEvaluator.db_executor_nodes.cache.sqlite_cache import SqliteCache
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SQLExecutorOutput


class TestSqliteCache:
    """Tests for the SqliteCache implementation with SQL output."""

    @pytest.fixture
    def cache_path(self, tmp_path):
        """Provides a temporary path for the SQLite cache file."""
        return str(tmp_path / "test_cache.db")

    @pytest.fixture
    def cache_instance(self, cache_path):
        """Provides a clean SqliteCache instance."""
        return SqliteCache(cache_path)

    def test_set_and_get_success(self, cache_instance, cache_path):
        """Verify that data can be saved and retrieved correctly."""
        table = SQLExecutorOutput(rows=[(1, "Alice"), (2, "Bob")])
        fetch_req = DataToFetch(
            db_path="users_db",
            query="SELECT * FROM users",
            dialect="sqlite"
        )

        data_to_cache = DataToCache[SQLExecutorOutput](
            **fetch_req.model_dump(exclude={'hash_key'}),
            result=table,
        )

        cache_instance.set_in_cache(cache_path, [data_to_cache])
        results = cache_instance.get_from_cache(cache_path, [fetch_req])

        assert len(results) == 1
        assert isinstance(results[0], DataToCache)
        assert results[0].result.rows == table.rows

    def test_cache_miss(self, cache_instance, cache_path):
        """Verify behavior when a hash_key is not found."""
        fetch_req = DataToFetch(
            db_path="users_db",
            query="SELECT *",
            dialect="sqlite"
        )
        results = cache_instance.get_from_cache(cache_path, [fetch_req])

        assert len(results) == 1
        assert isinstance(results[0], NotFoundInCacheError)

    def test_batch_operations(self, cache_instance, cache_path):
        """Verify multiple records can be handled at once."""
        table_a = SQLExecutorOutput(rows=[("A",)])
        table_b = SQLExecutorOutput(rows=[("B",)])

        data = [
            DataToCache(db_path="db", query="q1", result=table_a, dialect="sql"),
            DataToCache(db_path="db", query="q2", result=table_b, dialect="sql")
        ]

        cache_instance.set_in_cache(cache_path, data)

        fetches = [
            DataToFetch(db_path="db", query="q1", dialect="sql"),
            DataToFetch(db_path="db", query="q2", dialect="sql"),
            DataToFetch(db_path="db", query="q3", dialect="sql")
        ]
        results = cache_instance.get_from_cache(cache_path, fetches)

        assert len(results) == 3
        assert results[0].result.rows == [("A",)]
        assert results[1].result.rows == [("B",)]
        assert isinstance(results[2], NotFoundInCacheError)

    def test_duplicate_insert_keeps_first(self, cache_instance, cache_path):
        """INSERT OR IGNORE means the first cached value wins on duplicates."""
        fetch_req = DataToFetch(db_path="db", query="SELECT 1", dialect="sql")

        first = DataToCache(db_path="db", query="SELECT 1", result=SQLExecutorOutput(rows=[("first",)]), dialect="sql")
        second = DataToCache(db_path="db", query="SELECT 1", result=SQLExecutorOutput(rows=[("second",)]), dialect="sql")

        cache_instance.set_in_cache(cache_path, [first])
        cache_instance.set_in_cache(cache_path, [second])

        results = cache_instance.get_from_cache(cache_path, [fetch_req])
        assert results[0].result.rows == [("first",)]
