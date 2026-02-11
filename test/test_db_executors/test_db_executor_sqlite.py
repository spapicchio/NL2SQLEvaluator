import sqlite3
from unittest.mock import MagicMock

import pytest

from NL2SQLEvaluator.db_executor_nodes import SQLiteDBExecutor
from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import OutputTable, SQLCacheProtocol
from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import TaskToBeExecuted, ExecutorError


# Replace 'your_module' with the actual path to your SQLiteDBExecutor


class TestSQLiteDBExecutor:
    """Tests for the SQLiteDBExecutor using real database files."""

    @pytest.fixture
    def db_setup(self, tmp_path):
        """Creates a sample SQLite database."""
        db_path = tmp_path / "test_db.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO users (name) VALUES ('Alice'), ('Bob')")
        conn.commit()
        conn.close()
        return str(db_path)

    def test_execute_read_only_success(self, db_setup):
        """Tests a standard SELECT query in read-only mode."""
        executor = SQLiteDBExecutor()
        task = TaskToBeExecuted(
            db_path=db_setup,
            queries=["SELECT name FROM users ORDER BY name ASC"]
        )

        results = executor.execute_queries([task], allow_write=False)

        assert len(results) == 1
        assert isinstance(results[0][0], OutputTable)
        assert results[0][0].rows == [("Alice",), ("Bob",)]

    def test_execute_write_protection(self, db_setup):
        """Ensures that writing is blocked when allow_write is False."""
        executor = SQLiteDBExecutor()
        task = TaskToBeExecuted(
            db_path=db_setup,
            queries=["INSERT INTO users (name) VALUES ('Charlie')"]
        )

        results = executor.execute_queries([task], allow_write=False)

        # Should return an ExecutorError because PRAGMA query_only=ON is set
        assert isinstance(results[0][0], ExecutorError)
        assert "attempt to write a readonly database" in str(results[0][0]).lower()

    def test_execute_allow_write_success(self, db_setup):
        """Tests an INSERT query when allow_write is True."""
        executor = SQLiteDBExecutor()
        task = TaskToBeExecuted(
            db_path=db_setup,
            queries=["INSERT INTO users (name) VALUES ('Charlie')"]
        )

        results = executor.execute_queries([task], allow_write=True)

        assert isinstance(results[0][0], OutputTable)
        # The code returns rowcount for write operations
        assert results[0][0].rows == [([1])]

    def test_query_timeout(self, db_setup):
        """Tests that the executor kills long-running queries."""
        executor = SQLiteDBExecutor()
        # Create a recursive CTE that runs for a long time
        long_query = """
                     WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt)
                     SELECT x \
                     FROM cnt \
                     LIMIT 10000000 \
                     """
        task = TaskToBeExecuted(
            db_path=db_setup,
            queries=[long_query],
            timeout=0.1  # Very short timeout
        )

        results = executor.execute_queries([task])
        assert isinstance(results[0][0], ExecutorError)
        assert "Timeout" in str(results[0][0])

    def test_multiprocessing_batch(self, db_setup):
        """Tests handling multiple tasks and multiple queries at once."""
        executor = SQLiteDBExecutor()
        task1 = TaskToBeExecuted(db_path=db_setup, queries=["SELECT COUNT(*) FROM users"])
        task2 = TaskToBeExecuted(db_path=db_setup, queries=["SELECT name FROM users WHERE id=1"])

        results = executor.execute_queries([task1, task2], num_cpus=2)

        assert len(results) == 2
        assert results[0][0].rows == [(2,)]
        assert results[1][0].rows == [("Alice",)]

    def test_cache_hit(self, db_setup):
        """Tests that the executor returns cached results if available."""
        mock_cache = MagicMock(spec=SQLCacheProtocol)
        mock_output = OutputTable(rows=[("CachedName",)], executed_time=0.01)

        # Mocking the return value of get_from_cache
        # Note: In your code, it expects a list with an object that has a .result attribute
        mock_result = MagicMock()
        mock_result.result = mock_output
        mock_cache.get_from_cache.return_value = [mock_result]

        executor = SQLiteDBExecutor()
        task = TaskToBeExecuted(db_path=db_setup, queries=["SELECT name FROM users"])

        results = executor.execute_queries(
            [task],
            cache_db=mock_cache,
            cache_db_file="fake_cache.db"
        )

        assert results[0][0].rows == [("CachedName",)]
        mock_cache.get_from_cache.assert_called_once()

    def test_retrieve_preserves_request_order(self, tmp_path):
        """
        Ensures results are returned in the exact order of the input tasks and queries.

        Why: When using multiprocessing, results often finish out of order.
        We must verify our 'job_id' and 'idx' logic correctly maps them back.
        """
        # 1. Setup two different DBs to simulate multi-db tasks
        db1_path = str(tmp_path / "db1.sqlite")
        db2_path = str(tmp_path / "db2.sqlite")

        for path in [db1_path, db2_path]:
            with sqlite3.connect(path) as conn:
                conn.execute("CREATE TABLE test (val INTEGER)")

        executor = SQLiteDBExecutor()

        # 2. Define tasks with specific, identifiable expected results
        # Task 0: 3 queries
        task_0 = TaskToBeExecuted(
            db_path=db1_path,
            queries=[
                "SELECT 100",  # Expected: results[0][0]
                "SELECT 101",  # Expected: results[0][1]
                "SELECT 102"  # Expected: results[0][2]
            ]
        )
        # Task 1: 2 queries
        task_1 = TaskToBeExecuted(
            db_path=db2_path,
            queries=[
                "SELECT 200",  # Expected: results[1][0]
                "SELECT 201"  # Expected: results[1][1]
            ]
        )

        # 3. Execute
        all_tasks = [task_0, task_1]
        results = executor.execute_queries(all_tasks, num_cpus=4)

        # 4. Assertions for Structure
        assert len(results) == 2, "Should have 2 top-level result lists (one per task)"
        assert len(results[0]) == 3, "Task 0 should have 3 results"
        assert len(results[1]) == 2, "Task 1 should have 2 results"

        # 5. Assertions for Content Order
        assert results[0][0].rows == [(100,)]
        assert results[0][1].rows == [(101,)]
        assert results[0][2].rows == [(102,)]

        assert results[1][0].rows == [(200,)]
        assert results[1][1].rows == [(201,)]

    def test_query_with_special_characters(self, db_setup):
        """
        Verifies that the executor handles non-standard SQL formatting and Unicode.

        Why: LLM-generated SQL often contains unexpected characters, comments,
        or Unicode strings that can break string formatting or connection protocols.
        """
        executor = SQLiteDBExecutor()

        # We'll test:
        # 1. Newlines and tabs
        # 2. SQL Comments (inline and block)
        # 3. Unicode/Emojis in the data
        # 4. Single/Double quotes within the query
        special_queries = [
            "SELECT\nname\tFROM users\nWHERE name = 'Alice';",
            "SELECT name FROM users -- This is a comment",
            "SELECT '🚀' AS rocket_emoji",
            "SELECT \"name\" FROM users WHERE name = \"Bob\"",
            "/* Block comment */ SELECT 1"
        ]

        task = TaskToBeExecuted(
            db_path=db_setup,
            queries=special_queries
        )

        results = executor.execute_queries([task])

        # Assertions
        assert isinstance(results[0][0], OutputTable), "Should handle newlines/tabs"
        assert results[0][0].rows == [("Alice",)]

        assert isinstance(results[0][1], OutputTable), "Should handle trailing comments"
        assert results[0][1].rows == [("Alice",), ("Bob",)]

        assert results[0][2].rows == [("🚀",)], "Should handle Unicode/Emojis"

        assert isinstance(results[0][3], OutputTable), "Should handle double-quoted identifiers"
        assert results[0][3].rows == [("Bob",)]

        assert results[0][4].rows == [(1,)], "Should handle block comments"

        def test_execute_with_params(self, db_setup):
            """
            Verifies that parameterized queries correctly map and protect data.

            Why: Using 'params' prevents SQL injection and allows the same query
            structure to be reused with different values safely.
            """
            executor = SQLiteDBExecutor()

            # Test Case 1: Dictionary params (Named placeholders)
            # Test Case 2: List of dictionaries (mapping to multiple queries)
            queries = [
                "SELECT id FROM users WHERE name = :name",
                "SELECT name FROM users WHERE id = ?"
            ]

            # In TaskToBeExecuted, if params is a list, it must match queries length
            params_list = [
                {"name": "Alice"},  # For query 0
                (2,)  # For query 1 (SQLite also supports tuples for ?)
            ]

            task = TaskToBeExecuted(
                db_path=db_setup,
                queries=queries,
                params=params_list
            )

            results = executor.execute_queries([task])

            # Assertions
            assert results[0][0].rows == [(1,)], "Named parameters failed to match 'Alice' to ID 1"
            assert results[0][1].rows == [("Bob",)], "Positional parameters failed to match ID 2 to 'Bob'"

    def test_execute_with_params(self, db_setup):
        """
        Verifies that parameterized queries correctly map and protect data.

        Why: Using 'params' prevents SQL injection and allows the same query
        structure to be reused with different values safely.
        """
        executor = SQLiteDBExecutor()

        # Test Case 1: Dictionary params (Named placeholders)
        # Test Case 2: List of dictionaries (mapping to multiple queries)
        queries = [
            "SELECT id FROM users WHERE name = :name",
            "SELECT name FROM users WHERE id = ?",
            "SELECT name FROM users WHERE id = 1"
        ]

        # In TaskToBeExecuted, if params is a list, it must match queries length
        params_list = [
            {"name": "Alice"},  # For query 0
            (2,),
            ()
        ]

        task = TaskToBeExecuted(
            db_path=db_setup,
            queries=queries,
            params=params_list
        )

        results = executor.execute_queries([task])

        # Assertions
        assert results[0][0].rows == [(1,)], "Named parameters failed to match 'Alice' to ID 1"
        assert results[0][1].rows == [("Bob",)], "Positional parameters failed to match ID 2 to 'Bob'"
        assert results[0][1].rows == [("Bob",)], "Positional parameters failed to match ID 2 to 'Bob'"

    def test_params_sql_injection_safety(self, db_setup):
        """
        Ensures that malicious strings in params are treated as data, not code.
        """
        executor = SQLiteDBExecutor()

        # Malicious string that tries to 'break out' of the quote
        malicious_name = "'; DROP TABLE users; --"

        task = TaskToBeExecuted(
            db_path=db_setup,
            queries=["SELECT * FROM users WHERE name = ?"],
            params=[(malicious_name,)]
        )

        results = executor.execute_queries([task])

        # The query should simply return no results, NOT drop the table
        assert isinstance(results[0][0], OutputTable)
        assert len(results[0][0].rows) == 0

        # Verify table still exists
        verify_task = TaskToBeExecuted(db_path=db_setup, queries=["SELECT COUNT(*) FROM users"])
        verify_res = executor.execute_queries([verify_task])
        assert verify_res[0][0].rows[0][0] == 2