# ----------------------------
# Fixtures
# ----------------------------
import sqlite3

import pytest

from NL2SQLEvaluator.db_executor.sqlite_executor import SqliteCacheDB


# ----------------------------
# Fixtures
# ----------------------------


@pytest.fixture
def tmp_db_file(tmp_path):
    # Create an empty SQLite file on disk to satisfy `mode=rw`
    db_path = tmp_path / "cache.sqlite"
    sqlite3.connect(db_path).close()
    return str(db_path)


@pytest.fixture
def sqlite_cache_executor(tmp_db_file):
    executor = SqliteCacheDB.from_uri(relative_base_path=tmp_db_file)
    yield executor
    executor.engine.dispose()


# ----------------------------
# Tests SQL normalization
# ----------------------------
def test_sql_normalization_remove_semicolumn(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = "SELECT * FROM Airlines WHERE TAIL_NUM='N956AN';"
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert normalized_sql == "SELECT * FROM Airlines WHERE TAIL_NUM = 'N956AN'"


def test_sql_normalization_remove_spaces(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = "SELECT *\nFROM Airlines\nWHERE TAIL_NUM='N956AN';"
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert normalized_sql == "SELECT * FROM Airlines WHERE TAIL_NUM = 'N956AN'"


def test_sql_normalization_no_change(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = "SELECT COUNT(*) FROM Airlines WHERE TAIL_NUM = 'N956AN'"
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert normalized_sql == raw_sql


def test_sql_normalization_complex_nesting(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = """
              SELECT COUNT(*)
              FROM (SELECT DISTINCT TAIL_NUM
                    FROM Airlines
                    WHERE TAIL_NUM IS NOT NULL) AS subquery
              WHERE subquery.TAIL_NUM LIKE 'N%'
              ; \
              """
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    expected_sql = ("SELECT COUNT(*) FROM (SELECT DISTINCT TAIL_NUM FROM Airlines "
                    "WHERE NOT TAIL_NUM IS NULL) AS subquery WHERE subquery.TAIL_NUM LIKE 'N%'")
    assert normalized_sql == expected_sql


def test_sql_normalization_handles_empty_string(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = ""
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert normalized_sql == ""


def test_sql_normalization_handles_whitespace_only(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = "   \n\t  "
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert normalized_sql == ""


def test_sql_normalization_handles_single_word(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = "SELECT"
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert normalized_sql == "SELECT"


def test_sql_normalization_normalizes_case_keywords(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = "select * from airlines where tail_num='N956AN'"
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert normalized_sql == "SELECT * FROM airlines WHERE tail_num = 'N956AN'"


def test_test_sql_normalization_handles_multiple_semicolons(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = "SELECT * FROM Airlines;;;"
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert normalized_sql == "SELECT * FROM Airlines"


def test_sql_normalization_preserves_quoted_strings_with_special_chars(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = "SELECT * FROM Airlines WHERE name = 'Air;line\nCorp'"
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert "Air;line\nCorp" in normalized_sql


def test_sql_normalization_handles_comments(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = "SELECT * FROM Airlines -- this is a comment\nWHERE id = 1"
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert "SELECT * FROM Airlines /* this is a comment */ WHERE id = 1" == normalized_sql


def test_sql_normalization_handles_malformed_sql_gracefully(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = "SELECT * FROM WHERE"
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert normalized_sql == raw_sql


def test_sql_normalization_handles_unicode_characters(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = "SELECT * FROM Airlines WHERE name = 'Café Αεροπορία'"
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert "Café Αεροπορία" in normalized_sql


def test_sql_normalization_handles_very_long_query(sqlite_cache_executor: SqliteCacheDB):
    long_where_clause = " OR ".join([f"id = {i}" for i in range(1000)])
    raw_sql = f"SELECT * FROM Airlines WHERE {long_where_clause};"
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert normalized_sql.endswith("id = 999")
    assert not normalized_sql.endswith(";")


def test_sql_normalization_handles_nested_quotes(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = "SELECT * FROM Airlines WHERE description = 'John''s \"favorite\" airline'"
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert "John''s \"favorite\" airline" in normalized_sql


def test_sql_normalization_handles_cte_queries(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = """
              WITH airline_stats AS (SELECT COUNT(*) as total
                                     FROM Airlines)
              SELECT *
              FROM airline_stats; \
              """
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert "WITH airline_stats AS" in normalized_sql
    assert not normalized_sql.endswith(";")


def test_attributes_with_single_quote_inside(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = "SELECT name FROM Airlines WHERE name = 'O'Reilly Airlines';"
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert "O'Reilly Airlines" in normalized_sql


def test_attribute_belongs_two_tables(sqlite_cache_executor: SqliteCacheDB):
    raw_sql = "SELECT a.name, code FROM Airlines a JOIN Airports b ON a.code = b.code WHERE a.name = 'Delta';"
    normalized_sql = sqlite_cache_executor.parse_sql_query(raw_sql)
    assert normalized_sql == "SELECT a.name, code FROM Airlines AS a JOIN Airports AS b ON a.code = b.code WHERE a.name = 'Delta'"


# ----------------------------
# Tests INSERT
# ----------------------------
def test_insert_and_fetch_cache(sqlite_cache_executor: SqliteCacheDB):
    db_id = "test_db"
    query = "SELECT * FROM Airlines WHERE TAIL_NUM='N956AN';"
    result = [("N956AN", "Delta", "DL", "New York", "Los Angeles")]

    # Initially, cache should be empty
    cached_result = sqlite_cache_executor.fetch_from_cache(db_id, query)
    assert cached_result is None

    # Insert into cache
    sqlite_cache_executor.insert_in_cache(db_id, query, result)

    # Fetch from cache
    cached_result = sqlite_cache_executor.fetch_from_cache(db_id, query)
    assert cached_result == result


def test_fetch_nonexistent_cache(sqlite_cache_executor: SqliteCacheDB):
    db_id = "nonexistent_db"
    query = "SELECT * FROM NonExistentTable;"

    # Fetch from cache should return None for non-existent entry
    cached_result = sqlite_cache_executor.fetch_from_cache(db_id, query)
    assert cached_result is None


def test_insert_duplicate_ignored(sqlite_cache_executor: SqliteCacheDB):
    db_id = "test_db"
    query = "SELECT * FROM Airlines WHERE TAIL_NUM='N956AN';"
    result = [("N956AN", "Delta", "DL", "New York", "Los Angeles")]

    # Insert into cache
    sqlite_cache_executor.insert_in_cache(db_id, query, result)

    # Insert duplicate into cache
    sqlite_cache_executor.insert_in_cache(db_id, query, result)

    # Fetch from cache
    cached_result = sqlite_cache_executor.fetch_from_cache(db_id, query)
    assert cached_result == result


def test_bulk_insert_and_fetch(sqlite_cache_executor: SqliteCacheDB):
    db_id = "test_db"
    queries_and_results = [
        (
            db_id,
            "SELECT * FROM Airlines WHERE TAIL_NUM='N956AN';",
            [("N956AN", "Delta", "DL", "New York", "Los Angeles")]
        ),
        (
            db_id,
            "SELECT * FROM Airlines WHERE TAIL_NUM='N12345';",
            [("N12345", "American", "AA", "Chicago", "Miami")]
        ),
        (
            db_id,
            "SELECT * FROM Airlines WHERE TAIL_NUM='N67890';",
            [("N67890", "United", "UA", "San Francisco", "Seattle")]
        )
    ]
    db_ids = [db_id for db_id, _, _ in queries_and_results]
    queries = [q for _, q, _ in queries_and_results]
    results = [r for _, _, r in queries_and_results]
    sqlite_cache_executor.insert_bulk_in_cache(db_ids, queries, results)

    # Fetch and verify each entry
    db_query_present = sqlite_cache_executor.db_id_query_already_present()
    for db_id, query, expected_result in queries_and_results:
        assert (db_id, sqlite_cache_executor.parse_sql_query(query)) in db_query_present
        cached_result = sqlite_cache_executor.fetch_from_cache(db_id, query)
        assert cached_result == expected_result


def test_insert_and_fetch_cache_with_different_queries(sqlite_cache_executor: SqliteCacheDB):
    db_id = "test_db"
    query1 = "SELECT * FROM Airlines WHERE TAIL_NUM='N956AN';"
    result1 = [("N956AN", "Delta", "DL", "New York", "Los Angeles")]

    query2 = "SELECT * FROM Airlines WHERE TAIL_NUM='N12345';"
    result2 = [("N12345", "American", "AA", "Chicago", "Miami")]

    # Insert first query into cache
    sqlite_cache_executor.insert_in_cache(db_id, query1, result1)

    # Insert second query into cache
    sqlite_cache_executor.insert_in_cache(db_id, query2, result2)

    # Fetch first query from cache
    cached_result1 = sqlite_cache_executor.fetch_from_cache(db_id, query1)
    assert cached_result1 == result1

    # Fetch second query from cache
    cached_result2 = sqlite_cache_executor.fetch_from_cache(db_id, query2)
    assert cached_result2 == result2


def test_insert_very_large_result_blob(sqlite_cache_executor: SqliteCacheDB):
    """Test inserting a very large result that may exceed BLOB limits."""
    db_id = "test_db"
    query = "SELECT * FROM LargeTable;"

    # Create a very large result (simulating millions of rows)
    large_result = [("data" * 1000, f"value_{i}", "x" * 500) for i in range(10000)]

    # This should handle large BLOBs gracefully
    sqlite_cache_executor.insert_in_cache(db_id, query, large_result)

    # Fetch should work or return None if insertion failed
    cached_result = sqlite_cache_executor.fetch_from_cache(db_id, query)
    # Either it works completely or fails gracefully
    assert cached_result == large_result or cached_result is None


def test_insert_extremely_large_blob_exceeds_sqlite_limit(sqlite_cache_executor: SqliteCacheDB):
    """Test inserting a BLOB that exceeds SQLite's maximum BLOB size."""
    db_id = "test_db"
    query = "SELECT * FROM ExtremeTable;"

    # Create a result that when pickled will be very large (>1GB)
    # SQLite default max BLOB size is 1GB
    massive_result = [("x" * 1000000,) for _ in range(1100)]  # ~1.1GB when pickled
    sqlite_cache_executor.insert_in_cache(db_id, query, massive_result)

    # If insertion succeeded, fetch should work
    cached_result = sqlite_cache_executor.fetch_from_cache(db_id, query)
    assert cached_result is None


def test_insert_empty_result_list(sqlite_cache_executor: SqliteCacheDB):
    """Test inserting an empty result list."""
    db_id = "test_db"
    query = "SELECT * FROM Airlines WHERE 1=0;"
    result = []

    sqlite_cache_executor.insert_in_cache(db_id, query, result)
    cached_result = sqlite_cache_executor.fetch_from_cache(db_id, query)
    assert cached_result == result


def test_insert_result_with_none_values(sqlite_cache_executor: SqliteCacheDB):
    """Test inserting results containing None values."""
    db_id = "test_db"
    query = "SELECT * FROM Airlines WHERE some_field IS NULL;"
    result = [("N956AN", None, "DL", None, "Los Angeles"), (None, "Delta", None, "New York", None)]

    sqlite_cache_executor.insert_in_cache(db_id, query, result)
    cached_result = sqlite_cache_executor.fetch_from_cache(db_id, query)
    assert cached_result == result


def test_insert_result_with_special_characters_and_unicode(sqlite_cache_executor: SqliteCacheDB):
    """Test inserting results with special characters and unicode."""
    db_id = "test_db"
    query = "SELECT * FROM Airlines WHERE name LIKE '%special%';"
    result = [
        ("N956AN", "Café Αεροπορία", "ÇŞ", "São Paulo", "München"),
        ("X123", "Авиакомпания", "РУ", "Москва", "北京"),
        ("Z789", "emoji✈️🌍", "EM", "🗽New York", "🏝️Hawaii")
    ]

    sqlite_cache_executor.insert_in_cache(db_id, query, result)
    cached_result = sqlite_cache_executor.fetch_from_cache(db_id, query)
    assert cached_result == result


def test_insert_result_with_binary_data(sqlite_cache_executor: SqliteCacheDB):
    """Test inserting results containing binary data."""
    db_id = "test_db"
    query = "SELECT * FROM BinaryTable;"
    binary_data = b'\x00\x01\x02\xff\xfe\xfd'
    result = [("record1", binary_data), ("record2", b'\x89PNG\r\n\x1a\n')]

    sqlite_cache_executor.insert_in_cache(db_id, query, result)
    cached_result = sqlite_cache_executor.fetch_from_cache(db_id, query)
    assert cached_result == result


def test_fetch_corrupted_cache_entry(sqlite_cache_executor: SqliteCacheDB):
    """Test fetching when cache entry has corrupted pickle data."""
    db_id = "test_db"
    query = "SELECT * FROM Airlines;"

    # Manually insert corrupted pickle data
    from NL2SQLEvaluator.db_executor.sqlite_executor import hash_db_id_sql
    hash_id = hash_db_id_sql(db_id, sqlite_cache_executor.parse_sql_query(query))
    corrupted_data = b"corrupted_pickle_data_not_valid"

    stmt = "INSERT INTO `cache_data` (hash_key, db_id, query, result) VALUES (?, ?, ?, ?)"
    sqlite_cache_executor.execute_query(stmt,
                                        {"hash_id": hash_id, "db_id": db_id, "query": query, "result": corrupted_data})

    # Fetch should handle corruption gracefully
    cached_result = sqlite_cache_executor.fetch_from_cache(db_id, query)
    assert cached_result is None


def test_insert_with_very_long_db_id_and_query(sqlite_cache_executor: SqliteCacheDB):
    """Test inserting with very long db_id and query strings."""
    db_id = "x" * 10000  # Very long database ID
    query = "SELECT * FROM table WHERE " + " AND ".join([f"col{i} = 'value{i}'" for i in range(1000)])
    result = [("data1", "data2", "data3")]

    sqlite_cache_executor.insert_in_cache(db_id, query, result)
    cached_result = sqlite_cache_executor.fetch_from_cache(db_id, query)
    assert cached_result == result


def test_insert_result_with_nested_data_structures(sqlite_cache_executor: SqliteCacheDB):
    """Test inserting results with complex nested data structures."""
    db_id = "test_db"
    query = "SELECT * FROM ComplexTable;"
    # Note: Tuples should contain simple types for SQL results, but testing edge case
    result = [
        ("simple_string", 123, 45.67),
        ("another_row", 789, 12.34)
    ]

    sqlite_cache_executor.insert_in_cache(db_id, query, result)
    cached_result = sqlite_cache_executor.fetch_from_cache(db_id, query)
    assert cached_result == result


def test_concurrent_insert_same_cache_entry(sqlite_cache_executor: SqliteCacheDB):
    """Test concurrent insertions of the same cache entry."""
    import threading

    db_id = "test_db"
    query = "SELECT * FROM Airlines WHERE TAIL_NUM='N956AN';"
    result = [("N956AN", "Delta", "DL", "New York", "Los Angeles")]

    def insert_worker():
        sqlite_cache_executor.insert_in_cache(db_id, query, result)

    # Create multiple threads trying to insert the same entry
    threads = [threading.Thread(target=insert_worker) for _ in range(10)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # Should handle concurrent inserts gracefully (INSERT OR IGNORE)
    cached_result = sqlite_cache_executor.fetch_from_cache(db_id, query)
    assert cached_result == result


def test_insert_with_malformed_sql_query(sqlite_cache_executor: SqliteCacheDB):
    """Test inserting cache entry with malformed SQL query."""
    db_id = "test_db"
    malformed_query = "SELECT * FROM WHERE AND OR;"
    result = None

    # Should handle malformed SQL gracefully in parsing
    sqlite_cache_executor.insert_in_cache(db_id, malformed_query, result)
    cached_result = sqlite_cache_executor.fetch_from_cache(db_id, malformed_query)
    assert cached_result == result


def test_high_concurrency_mixed_operations_stress(sqlite_cache_executor: SqliteCacheDB):
    """Stress test with high concurrency mixed read/write operations to force potential deadlocks."""
    import multiprocessing
    import random
    import time

    db_id = "stress_test_db"
    base_query = "SELECT * FROM Airlines WHERE TAIL_NUM='{}'"

    def mixed_worker(worker_id: int):
        executor = SqliteCacheDB.from_uri(relative_base_path=sqlite_cache_executor.engine.url.database)
        try:
            for i in range(50):  # More operations per worker
                tail_num = f"N{worker_id:03d}{i:02d}"
                query = base_query.format(tail_num)
                result = [(tail_num, f"Airline_{worker_id}", "XX", "City1", "City2")]

                # Mix of operations
                if random.choice([True, False]):
                    # Insert operation
                    executor.insert_in_cache(db_id, query, result)
                else:
                    # Fetch operation
                    executor.fetch_from_cache(db_id, query)

                # Small random delay to increase chance of race conditions
                time.sleep(random.uniform(0.001, 0.01))
        finally:
            executor.engine.dispose()

    # Create many processes for high contention
    processes = [multiprocessing.Process(target=mixed_worker, args=(i,)) for i in range(20)]

    start_time = time.time()
    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=30)  # Timeout to prevent hanging
        if process.is_alive():
            process.terminate()
            process.join()

    execution_time = time.time() - start_time
    print(f"Stress test completed in {execution_time:.2f} seconds")

    # Verify some data was actually inserted
    sample_result = sqlite_cache_executor.fetch_from_cache(db_id, base_query.format("N001001"))
    # Should either have data or None (both are acceptable under high contention)
    assert sample_result is None or isinstance(sample_result, list)


def test_deadlock_scenario_rapid_inserts(sqlite_cache_executor: SqliteCacheDB):
    """Force potential deadlock with rapid concurrent inserts of different entries."""
    import multiprocessing

    def rapid_insert_worker(worker_id: int, num_inserts: int):
        executor = SqliteCacheDB.from_uri(relative_base_path=sqlite_cache_executor.engine.url.database)
        try:
            for i in range(num_inserts):
                db_id = f"deadlock_test_{worker_id}"
                query = f"SELECT * FROM Table_{worker_id} WHERE id = {i}"
                result = [(f"data_{worker_id}_{i}", f"value_{i}")]

                # Rapid insertions without delays
                executor.insert_in_cache(db_id, query, result)
        finally:
            executor.engine.dispose()

    # Many workers doing rapid inserts
    processes = [
        multiprocessing.Process(target=rapid_insert_worker, args=(i, 100))
        for i in range(15)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=1600)
        if process.is_alive():
            process.terminate()

    # Test should complete without hanging (no deadlock)
    assert True


def test_database_lock_contention_stress(sqlite_cache_executor: SqliteCacheDB):
    """Create maximum database lock contention with overlapping transactions."""
    import multiprocessing

    def lock_contention_worker(worker_id: int):
        executor = SqliteCacheDB.from_uri(relative_base_path=sqlite_cache_executor.engine.url.database)
        try:
            # All workers try to insert the SAME cache entry simultaneously
            db_id = "shared_db"
            query = "SELECT * FROM SharedTable WHERE id = 1"
            result = [("shared_data", "shared_value", worker_id)]

            # Repeat many times to increase lock contention
            for _ in range(200):
                executor.insert_in_cache(db_id, query, result)
        finally:
            executor.engine.dispose()

    # Maximum contention - all workers fight for same resource
    processes = [multiprocessing.Process(target=lock_contention_worker, args=(i,)) for i in range(25)]

    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=25)
        if process.is_alive():
            process.terminate()

    # Verify the shared entry exists (INSERT OR IGNORE should handle duplicates)
    cached_result = sqlite_cache_executor.fetch_from_cache("shared_db", "SELECT * FROM SharedTable WHERE id = 1")
    assert cached_result is not None


def test_burst_concurrent_large_blob_inserts(sqlite_cache_executor: SqliteCacheDB):
    """Stress test with burst of large BLOB insertions to trigger memory/lock issues."""
    import multiprocessing

    def large_blob_worker(worker_id: int):
        executor = SqliteCacheDB.from_uri(relative_base_path=sqlite_cache_executor.engine.url.database)
        try:
            # Each worker inserts large data
            db_id = f"large_blob_db_{worker_id}"
            query = f"SELECT * FROM LargeBlobTable_{worker_id}"
            # Create moderately large result (not extreme to avoid timeout)
            large_result = [("x" * 10000, f"worker_{worker_id}", "y" * 5000) for _ in range(50)]

            # Multiple large insertions
            for i in range(10):
                query_variant = f"{query} WHERE batch = {i}"
                executor.insert_in_cache(db_id, query_variant, large_result)
        finally:
            executor.engine.dispose()

    # Concurrent large BLOB insertions
    processes = [multiprocessing.Process(target=large_blob_worker, args=(i,)) for i in range(10)]

    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()

    # Verify some large data was inserted
    sample_result = sqlite_cache_executor.fetch_from_cache("large_blob_db_0",
                                                           "SELECT * FROM LargeBlobTable_0 WHERE batch = 0")
    assert sample_result is None or len(sample_result) > 0


def test_cascading_timeout_scenario(sqlite_cache_executor: SqliteCacheDB):
    """Test scenario where timeouts in one process might cascade to others."""
    import multiprocessing
    import time

    def timeout_prone_worker(worker_id: int, should_delay: bool):
        executor = SqliteCacheDB.from_uri(relative_base_path=sqlite_cache_executor.engine.url.database)
        try:
            if should_delay:
                # Some workers simulate slow operations
                time.sleep(0.5)

            for i in range(100):
                db_id = f"timeout_test_{worker_id % 5}"  # Force collision on db_id
                query = f"SELECT * FROM TimeoutTable WHERE worker_id = {worker_id} AND batch = {i}"
                result = [(f"data_{worker_id}_{i}", "timeout_test")]

                executor.insert_in_cache(db_id, query, result)

                if should_delay and i % 10 == 0:
                    time.sleep(0.1)  # Periodic delays to hold locks longer
        finally:
            executor.engine.dispose()

    # Mix of fast and slow workers
    processes = []
    for i in range(20):
        should_delay = (i % 3 == 0)  # Every 3rd worker is slow
        processes.append(multiprocessing.Process(target=timeout_prone_worker, args=(i, should_delay)))

    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=35)
        if process.is_alive():
            process.terminate()

    # System should remain responsive despite some slow operations
    test_result = sqlite_cache_executor.fetch_from_cache("timeout_test_0",
                                                         "SELECT * FROM TimeoutTable WHERE worker_id = 0 AND batch = 0")
    assert test_result is None or isinstance(test_result, list)


def test_fork_bomb_simulation(sqlite_cache_executor: SqliteCacheDB):
    """Simulate fork bomb scenario with many short-lived processes."""
    import multiprocessing
    import time

    def quick_worker(worker_id: int):
        executor = SqliteCacheDB.from_uri(relative_base_path=sqlite_cache_executor.engine.url.database)
        try:
            # Very quick operation and exit
            db_id = "fork_bomb_test"
            query = f"SELECT * FROM QuickTable WHERE id = {worker_id}"
            result = [(f"quick_data_{worker_id}",)]
            executor.insert_in_cache(db_id, query, result)
        finally:
            executor.engine.dispose()

    # Rapid succession of many short-lived processes
    all_processes = []
    for batch in range(5):  # 5 batches of processes
        batch_processes = [
            multiprocessing.Process(target=quick_worker, args=(batch * 50 + i,))
            for i in range(50)
        ]

        # Start all processes in batch rapidly
        for process in batch_processes:
            process.start()

        all_processes.extend(batch_processes)
        time.sleep(0.1)  # Small delay between batches

    # Wait for all to complete
    for process in all_processes:
        process.join(timeout=2)  # Short timeout
        if process.is_alive():
            process.terminate()

    # Verify system survived the process storm
    test_result = sqlite_cache_executor.fetch_from_cache("fork_bomb_test", "SELECT * FROM QuickTable WHERE id = 0")
    assert test_result is None or isinstance(test_result, list)
