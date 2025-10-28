import pytest

from NL2SQLEvaluator.db_executor_nodes import SqliteCache, SQLiteDBExecutor
from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecuteTask
from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import NotFoundInCacheError, OutputTable, DataToFetch, \
    DataToCache


@pytest.fixture
def db_file(tmp_path):
    """
    Create a temporary sqlite database file under tmp_path, populate it with sample rows,
    and yield the file path as a string.
    """
    db_file = tmp_path / "test.db"
    yield str(db_file)


@pytest.fixture
def cache_db(db_file) -> SqliteCache:
    return SqliteCache()


def return_rows_in_cache(db_file: str) -> OutputTable:
    executor = SQLiteDBExecutor()
    task = ExecuteTask(
        db_files=[db_file],
        queries=["SELECT * FROM `cache_data`"],
    )
    result = executor.execute_queries(
        tasks=[task]
    )[0][0]
    return result


class TestSqliteCache:
    def test_insert_works(self, cache_db: SqliteCache, db_file):
        query = "SELECT * FROM users WHERE age > 30"
        result = OutputTable(rows=[("Alice", 35), ("Bob", 40)])
        cache_db.set_in_cache(
            db_file,
            data_to_cache=[DataToCache(db_id='test', query=query, result=result)],
        )

        rows = return_rows_in_cache(db_file).rows
        assert len(rows) == 1
        assert rows[0][1] == "test"
        assert rows[0][2] == query
        assert rows[0][3] == result.compress()

    def test_cache_and_retrieve(self, cache_db: SqliteCache, db_file):
        query = "SELECT * FROM users WHERE age > 30"
        result = OutputTable(rows=[("Carol", 40)])
        cache_db.set_in_cache(
            db_file,
            data_to_cache=[DataToCache(db_id='test', query=query, result=result)],
        )

        retrieved_result = cache_db.get_from_cache(db_file, data_to_fetch=[DataToFetch(db_id='test', query=query)])[0]
        assert retrieved_result == result

    def test_multiple_inserts(self, cache_db: SqliteCache, db_file):
        queries_and_results = [
            ("SELECT * FROM users WHERE age > 30", OutputTable(rows=[("Dave", 45)])),
            ("SELECT * FROM users WHERE age < 20", OutputTable(rows=[("Eve", 18)])),
        ]

        data_to_cache = [
            DataToCache(db_id='test', query=query, result=result)
            for query, result in queries_and_results
        ]
        cache_db.set_in_cache(db_file, data_to_cache=data_to_cache)

        for query, expected_result in queries_and_results:
            retrieved_result = cache_db.get_from_cache(db_file, data_to_fetch=[DataToFetch(db_id='test', query=query)])[
                0]
            assert retrieved_result == expected_result

    def test_cache_miss(self, cache_db: SqliteCache, db_file):
        query = "SELECT * FROM users WHERE age > 50"
        retrieved_result = cache_db.get_from_cache(db_file, data_to_fetch=[DataToFetch(db_id='test', query=query)])
        assert isinstance(retrieved_result[0], NotFoundInCacheError)

    def test_multiple_cache_misses(self, cache_db: SqliteCache, db_file):
        queries = [
            "SELECT * FROM users WHERE age > 60",
            "SELECT * FROM users WHERE age < 10",
        ]
        data_to_fetch = [DataToFetch(db_id='test', query=query) for query in queries]
        retrieved_results = cache_db.get_from_cache(db_file, data_to_fetch=data_to_fetch)

        for result in retrieved_results:
            assert isinstance(result, NotFoundInCacheError)

    def test_mixed_cache_hits_and_misses(self, cache_db: SqliteCache, db_file):
        # Insert one query into the cache
        hit_query = "SELECT * FROM users WHERE age > 30"
        hit_result = OutputTable(rows=[("Frank", 55)])
        cache_db.set_in_cache(
            db_file,
            data_to_cache=[DataToCache(db_id='test', query=hit_query, result=hit_result)],
        )

        # Prepare one hit and one miss
        miss_query = "SELECT * FROM users WHERE age < 10"
        data_to_fetch = [
            DataToFetch(db_id='test', query=hit_query),
            DataToFetch(db_id='test', query=miss_query),
        ]

        retrieved_results = cache_db.get_from_cache(db_file, data_to_fetch=data_to_fetch)

        assert retrieved_results[0] == hit_result
        assert isinstance(retrieved_results[1], NotFoundInCacheError)

    def test_duplicate_insert_ignored(self, cache_db: SqliteCache, db_file):
        query = "SELECT duplicate_test"
        result = OutputTable(rows=[("Dup", 1)])
        data = DataToCache(db_id='test', query=query, result=result)

        cache_db.set_in_cache(db_file, data_to_cache=[data])
        cache_db.set_in_cache(db_file, data_to_cache=[data])  # same item again

        rows = return_rows_in_cache(db_file).rows
        assert len(rows) == 1

    def test_empty_result_cached_and_retrieved(self, cache_db: SqliteCache, db_file):
        query = "SELECT empty_result"
        empty_result = OutputTable(rows=[])
        cache_db.set_in_cache(db_file, data_to_cache=[DataToCache(db_id='test', query=query, result=empty_result)])

        retrieved = cache_db.get_from_cache(db_file, data_to_fetch=[DataToFetch(db_id='test', query=query)])[0]
        assert retrieved == empty_result

    def test_large_result(self, cache_db: SqliteCache, db_file):
        query = "SELECT large_result"
        large_rows = [(f"name_{i}", i) for i in range(500)]
        large_result = OutputTable(rows=large_rows)
        cache_db.set_in_cache(db_file, data_to_cache=[DataToCache(db_id='test', query=query, result=large_result)])

        retrieved = cache_db.get_from_cache(db_file, data_to_fetch=[DataToFetch(db_id='test', query=query)])[0]
        assert retrieved == large_result

    def test_retrieve_preserves_request_order(self, cache_db: SqliteCache, db_file):
        q1 = "SELECT order_test_1"
        r1 = OutputTable(rows=[("One", 1)])
        q2 = "SELECT order_test_2"
        r2 = OutputTable(rows=[("Two", 2)])

        cache_db.set_in_cache(db_file, data_to_cache=[
            DataToCache(db_id='test', query=q1, result=r1),
            DataToCache(db_id='test', query=q2, result=r2),
        ])

        # Request in reverse order and ensure results preserve the request order
        fetched = cache_db.get_from_cache(db_file, data_to_fetch=[
            DataToFetch(db_id='test', query=q2),
            DataToFetch(db_id='test', query=q1),
        ])
        assert fetched[0] == r2
        assert fetched[1] == r1

    def test_query_with_special_characters(self, cache_db: SqliteCache, db_file):
        query = "SELECT * FROM users WHERE name = 'O''Reilly' AND note = 'emoji: 🚀\\nnew'"
        result = OutputTable(rows=[("O'Reilly", "emoji: 🚀\nnew")])
        cache_db.set_in_cache(db_file, data_to_cache=[DataToCache(db_id='test', query=query, result=result)])

        retrieved = cache_db.get_from_cache(db_file, data_to_fetch=[DataToFetch(db_id='test', query=query)])[0]
        assert retrieved == result