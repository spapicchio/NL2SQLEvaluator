"""
Tests for Neo4jDBExecutor.

Uses mocked Neo4j driver — no real Neo4j instance required.
"""
from unittest.mock import MagicMock, patch

import pytest

from NL2SQLEvaluator.db_executor_nodes.db_executor_input import TaskToBeExecuted
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import CypherExecutorOutput, ExecutorError
from NL2SQLEvaluator.db_executor_nodes.neo4j_db_executor import (
    Neo4jDBExecutor, _execute_single_cypher, _parse_bolt_uri, _driver_cache, _driver_lock
)


@pytest.fixture(autouse=True)
def clear_driver_cache():
    """Clear driver cache between tests."""
    with _driver_lock:
        _driver_cache.clear()
    yield
    with _driver_lock:
        _driver_cache.clear()


class TestUriParsing:

    def test_full_uri(self):
        result = _parse_bolt_uri("bolt://admin:secret@myhost:7688/mydb")
        assert result["host"] == "myhost"
        assert result["port"] == 7688
        assert result["username"] == "admin"
        assert result["password"] == "secret"
        assert result["database"] == "mydb"
        assert result["bolt_url"] == "bolt://myhost:7688"

    def test_minimal_uri(self):
        result = _parse_bolt_uri("bolt://localhost:7687")
        assert result["host"] == "localhost"
        assert result["port"] == 7687
        assert result["username"] is None
        assert result["password"] is None
        assert result["database"] is None

    def test_uri_with_auth_no_db(self):
        result = _parse_bolt_uri("bolt://user:pass@host:7687")
        assert result["username"] == "user"
        assert result["password"] == "pass"
        assert result["database"] is None


class TestNeo4jDBExecutor:

    @pytest.fixture
    def executor(self):
        return Neo4jDBExecutor()

    @patch("NL2SQLEvaluator.db_executor_nodes.neo4j_db_executor._get_or_create_driver")
    def test_execute_queries_success(self, mock_get_driver, executor):
        """Mock driver returns records, verify CypherExecutorOutput."""
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_driver.return_value = mock_driver

        # execute_read returns the result of the tx function
        mock_session.execute_read.return_value = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]

        task = TaskToBeExecuted(
            db_path="bolt://localhost:7687/testdb",
            queries=["MATCH (n:Person) RETURN n.name AS name, n.age AS age"],
            timeout=10
        )

        results = executor.execute_queries([task])

        assert len(results) == 1
        assert len(results[0]) == 1
        result = results[0][0]
        assert isinstance(result, CypherExecutorOutput)
        # Keys sorted alphabetically: "age", "name"
        assert len(result.rows) == 2
        assert result.rows[0] == (30, "Alice")
        assert result.rows[1] == (25, "Bob")

    @patch("NL2SQLEvaluator.db_executor_nodes.neo4j_db_executor._get_or_create_driver")
    def test_read_only_enforcement(self, mock_get_driver, executor):
        """Verify session.execute_read() is used (not execute_write)."""
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_driver.return_value = mock_driver

        mock_session.execute_read.return_value = []

        task = TaskToBeExecuted(
            db_path="bolt://localhost:7687/testdb",
            queries=["MATCH (n) RETURN n"],
            timeout=10
        )

        executor.execute_queries([task])

        mock_session.execute_read.assert_called_once()
        mock_session.execute_write.assert_not_called()

    @patch("NL2SQLEvaluator.db_executor_nodes.neo4j_db_executor._get_or_create_driver")
    def test_bad_cypher_resilient(self, mock_get_driver, executor):
        """Syntax errors return ExecutorError without crashing the batch."""
        import neo4j.exceptions
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_driver.return_value = mock_driver

        mock_session.execute_read.side_effect = neo4j.exceptions.CypherSyntaxError("Invalid syntax")

        task = TaskToBeExecuted(
            db_path="bolt://localhost:7687/testdb",
            queries=["INVALID CYPHER STATEMENT"],
            timeout=10
        )

        results = executor.execute_queries([task])

        assert isinstance(results[0][0], ExecutorError)
        assert "Invalid syntax" in str(results[0][0])

    def test_cache_hit_prevents_execution(self, executor):
        """Mock cache hit, verify driver not called."""
        mock_cache = MagicMock()
        cache_file = "cache.db"

        fake_table = CypherExecutorOutput(rows=[("CachedData",)], execution_time=0.01)
        mock_cache.get_from_cache.return_value = [
            MagicMock(result=fake_table)
        ]

        task = TaskToBeExecuted(
            db_path="bolt://localhost:7687/testdb",
            queries=["MATCH (n) RETURN n"],
            timeout=10
        )

        with patch("NL2SQLEvaluator.db_executor_nodes.neo4j_db_executor._execute_single_cypher") as mock_worker:
            results = executor.execute_queries(
                [task], cache_db=mock_cache, cache_db_file=cache_file
            )

            assert results[0][0].rows == [("CachedData",)]
            mock_worker.assert_not_called()
            mock_cache.get_from_cache.assert_called_once()

    @patch("NL2SQLEvaluator.db_executor_nodes.neo4j_db_executor._get_or_create_driver")
    def test_timeout_handling(self, mock_get_driver, executor):
        """Mock timeout exception."""
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_driver.return_value = mock_driver

        mock_session.execute_read.side_effect = Exception("Transaction timed out")

        task = TaskToBeExecuted(
            db_path="bolt://localhost:7687/testdb",
            queries=["MATCH (n) RETURN n"],
            timeout=1
        )

        results = executor.execute_queries([task])

        assert isinstance(results[0][0], ExecutorError)
        assert "Timeout" in str(results[0][0])

    @patch("NL2SQLEvaluator.db_executor_nodes.neo4j_db_executor._get_or_create_driver")
    def test_parallel_execution(self, mock_get_driver, executor):
        """Multiple tasks processed concurrently."""
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_driver.return_value = mock_driver

        mock_session.execute_read.return_value = [{"val": 42}]

        tasks = [
            TaskToBeExecuted(
                db_path="bolt://localhost:7687/testdb",
                queries=[f"MATCH (n) WHERE n.id = {i} RETURN n.val AS val"],
                timeout=10
            )
            for i in range(5)
        ]

        results = executor.execute_queries(tasks, max_workers=3)

        assert len(results) == 5
        for task_results in results:
            assert isinstance(task_results[0], CypherExecutorOutput)
            assert task_results[0].rows == [(42,)]
