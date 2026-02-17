"""
Tests for SQLiteDBExecutor.

Why: Ensures parallel execution, SQLite safety constraints (query_only), 
and cache integration work as expected.
"""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from NL2SQLEvaluator.db_executor_nodes.db_executor_input import ExecutorError, TaskToBeExecuted
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SQLExecutorOutput
from NL2SQLEvaluator.db_executor_nodes.sqlite_db_executor import (
    SQLiteDBExecutor, _execute_single_query
)


class TestSQLiteDBExecutor:

    @pytest.fixture
    def db_path(self, tmp_path):
        """Creates a real SQLite DB for integration testing."""
        path = tmp_path / "test_db.sqlite"
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')")
        return str(path)

    @pytest.fixture
    def executor(self):
        return SQLiteDBExecutor()

    def test_execute_single_query_read_only(self, db_path):
        """Tests that query_only=ON prevents writes during read tasks."""
        # job_id, idx, db_file, query, timeout, allow_write, params
        _, _, result = _execute_single_query(
            0, 0, db_path, "INSERT INTO users VALUES (3, 'Charlie')", 5.0, False, {}
        )

        assert isinstance(result, ExecutorError)
        assert "attempt to write a readonly database" in str(result).lower()

    def test_execute_queries(self, executor: SQLiteDBExecutor, db_path: str):
        """Tests that query_only=ON prevents writes during read tasks."""
        # job_id, idx, db_file, query, timeout, allow_write, params
        result = executor.execute_queries(
            tasks=[
                TaskToBeExecuted(
                    db_path=db_path,
                    queries=["SELECT name FROM users"],
                    timeout=[500]
                )
            ],
        )
        target = SQLExecutorOutput(rows=[("Bob",), ("Alice",)])
        assert not isinstance(result[0][0], ExecutorError)
        assert result[0][0].is_equivalent_to(target, is_row_order_important=False)

    def test_bad_sql_resilient(self, executor: SQLiteDBExecutor, db_path: str):
        res = executor.execute_queries(
            tasks=[
                TaskToBeExecuted(
                    db_path=db_path,
                    queries=[
                        "SELECT * FROM non_existent_table",
                        "SELECT name FROM users",
                        "SELECT name FROM users ORDER BY id DESC",
                        "MALFORMED SQL STATEMENT",
                    ],
                    timeout=5
                )
            ],
        )

        target = SQLExecutorOutput(rows=[("Alice",), ("Bob",)])
        assert isinstance(res[0][0], ExecutorError)
        assert not isinstance(res[0][1], ExecutorError) and res[0][1].is_equivalent_to(target)
        target = SQLExecutorOutput(rows=[("Bob",), ("Alice",)])
        assert not isinstance(res[0][2], ExecutorError) \
               and res[0][2].is_equivalent_to(target, is_row_order_important=True)
        assert isinstance(res[0][3], ExecutorError)

    def test_execute_queries_parallel_success(self, executor, db_path):
        """Tests batch execution of multiple queries across tasks."""
        tasks = [
            TaskToBeExecuted(db_path=db_path, queries=["SELECT name FROM users WHERE id=1"]),
            TaskToBeExecuted(db_path=db_path, queries=["SELECT COUNT(*) FROM users"])
        ]

        results = executor.execute_queries(tasks, num_cpus=2)

        assert len(results) == 2
        assert results[0][0].rows == [("Alice",)]
        assert results[1][0].rows == [(2,)]

    def test_cache_hit_prevents_execution(self, executor, db_path):
        """Verifies that a cache hit skips the SQL execution step."""
        mock_cache = MagicMock()
        cache_file = "cache.db"

        # Setup mock to return a hit
        fake_table = SQLExecutorOutput(rows=[("CachedData",)], execution_time=0.01)
        mock_cache.get_from_cache.return_value = [
            MagicMock(result=fake_table)  # Mocking DataToCache object
        ]
        task = TaskToBeExecuted(db_path=db_path, queries=["SELECT * FROM users"])

        # Patch _execute_single_query to ensure it's NEVER called
        with patch("NL2SQLEvaluator.db_executor_nodes.sqlite_db_executor._execute_single_query") as mock_worker:
            results = executor.execute_queries(
                [task], cache_db=mock_cache, cache_db_file=cache_file
            )

            assert results[0][0].rows == [("CachedData",)]
            mock_worker.assert_not_called()
            mock_cache.get_from_cache.assert_called_once()

    def test_timeout_handling(self, executor, db_path):
        """Force func_timeout to raise FunctionTimedOut so we don't rely on slow SQL tricks.
        Please note that the timeout cannot be tested with a super small timeout because otherwise the sqlite engine will not even be called        """
        from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecutorError
        results = executor.execute_queries(
            tasks=[
                TaskToBeExecuted(
                    db_path=db_path,
                    queries=[
                        """ WITH RECURSIVE cnt(x) AS (SELECT 1
                                                      UNION ALL
                                                      SELECT x + 1
                                                      FROM cnt
                                                      WHERE x < 10e10)
                            SELECT x
                            FROM cnt;
                        """
                    ],
                    timeout=3
                ),
            ],
        )
        assert isinstance(results[0][0], ExecutorError)
        assert "Timeout" in str(results[0][0])

    def test_multiple_timeout(self, executor: SQLiteDBExecutor, db_path: str):
        """
        Force func_timeout to raise FunctionTimedOut so we don't rely on slow SQL tricks.
        """
        from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecutorError
        res = executor.execute_queries(
            tasks=[
                TaskToBeExecuted(
                    db_path=db_path,
                    queries=[
                                """ WITH RECURSIVE cnt(x) AS (SELECT 1
                                                              UNION ALL
                                                              SELECT x + 1
                                                              FROM cnt
                                                              WHERE x < 10e10)
                                    SELECT x
                                    FROM cnt;
                                """
                            ] * 5,
                    timeout=3,
                )
            ],
        )
        assert isinstance(res[0][0], ExecutorError)
        assert isinstance(res[0][1], ExecutorError)
        assert isinstance(res[0][2], ExecutorError)
        assert isinstance(res[0][3], ExecutorError)
        assert isinstance(res[0][4], ExecutorError)

    def test_allow_write_behavior(self, executor, db_path):
        """Tests that rowcount is captured correctly for write operations."""
        query = "UPDATE users SET name='Updated' WHERE id=1"
        task = TaskToBeExecuted(db_path=db_path, queries=[query])
        results = executor.execute_queries([task], allow_write=True)

        # Your implementation returns rows=[(affected,)] for writes
        assert results[0][0].rows == [(1,)]
