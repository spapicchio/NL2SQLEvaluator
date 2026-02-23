"""
Tests for SparqlEndpointExecutor.

Uses mocked SPARQLWrapper — no real SPARQL endpoint required.
"""
from unittest.mock import MagicMock, patch

import pytest

from NL2SQLEvaluator.db_executor_nodes.db_executor_input import TaskToBeExecuted
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SparqlExecutorOutput, ExecutorError
from NL2SQLEvaluator.db_executor_nodes.sparql_endpoint_executor import (
    SparqlEndpointExecutor, _execute_single_sparql, _check_read_only
)


class TestSparqlEndpointExecutor:

    @pytest.fixture
    def executor(self):
        return SparqlEndpointExecutor()

    @patch("NL2SQLEvaluator.db_executor_nodes.sparql_endpoint_executor.SPARQLWrapper")
    def test_execute_queries_success(self, MockSPARQLWrapper, executor):
        """Mock endpoint returns bindings, verify SparqlExecutorOutput."""
        mock_instance = MagicMock()
        MockSPARQLWrapper.return_value = mock_instance

        mock_instance.query.return_value.convert.return_value = {
            "results": {
                "bindings": [
                    {"name": {"type": "literal", "value": "Alice"}, "age": {"type": "literal", "value": "30"}},
                    {"name": {"type": "literal", "value": "Bob"}, "age": {"type": "literal", "value": "25"}},
                ]
            }
        }

        task = TaskToBeExecuted(
            db_path="http://localhost:8890/sparql",
            queries=["SELECT ?name ?age WHERE { ?s foaf:name ?name ; foaf:age ?age }"],
            timeout=30
        )

        results = executor.execute_queries([task])

        assert len(results) == 1
        assert len(results[0]) == 1
        result = results[0][0]
        assert isinstance(result, SparqlExecutorOutput)
        # Keys sorted: "age", "name"
        assert len(result.rows) == 2
        assert result.rows[0] == ("30", "Alice")
        assert result.rows[1] == ("25", "Bob")

    def test_read_only_enforcement(self):
        """Verify write queries are rejected."""
        with pytest.raises(ExecutorError, match="Write operations"):
            _check_read_only("INSERT DATA { <s> <p> <o> }")

        with pytest.raises(ExecutorError, match="Write operations"):
            _check_read_only("DELETE WHERE { ?s ?p ?o }")

        with pytest.raises(ExecutorError, match="Write operations"):
            _check_read_only("DROP GRAPH <http://example.org/>")

        # SELECT should be fine
        _check_read_only("SELECT ?s WHERE { ?s ?p ?o }")

    @patch("NL2SQLEvaluator.db_executor_nodes.sparql_endpoint_executor.SPARQLWrapper")
    def test_bad_query_resilient(self, MockSPARQLWrapper, executor):
        """Query errors return ExecutorError without crashing the batch."""
        mock_instance = MagicMock()
        MockSPARQLWrapper.return_value = mock_instance
        mock_instance.query.side_effect = Exception("SPARQL parse error")

        task = TaskToBeExecuted(
            db_path="http://localhost:8890/sparql",
            queries=["INVALID SPARQL"],
            timeout=30
        )

        results = executor.execute_queries([task])

        assert isinstance(results[0][0], ExecutorError)
        assert "SPARQL parse error" in str(results[0][0])

    def test_cache_hit_prevents_execution(self, executor):
        """Mock cache hit, verify endpoint not called."""
        mock_cache = MagicMock()
        cache_file = "cache.db"

        fake_table = SparqlExecutorOutput(rows=[("CachedData",)], execution_time=0.01)
        mock_cache.get_from_cache.return_value = [
            MagicMock(result=fake_table)
        ]

        task = TaskToBeExecuted(
            db_path="http://localhost:8890/sparql",
            queries=["SELECT ?s WHERE { ?s ?p ?o }"],
            timeout=30
        )

        with patch("NL2SQLEvaluator.db_executor_nodes.sparql_endpoint_executor._execute_single_sparql") as mock_worker:
            results = executor.execute_queries(
                [task], cache_db=mock_cache, cache_db_file=cache_file
            )

            assert results[0][0].rows == [("CachedData",)]
            mock_worker.assert_not_called()
            mock_cache.get_from_cache.assert_called_once()

    @patch("NL2SQLEvaluator.db_executor_nodes.sparql_endpoint_executor.func_timeout")
    def test_timeout_handling(self, mock_func_timeout, executor):
        """Mock timeout exception."""
        from func_timeout import FunctionTimedOut
        mock_func_timeout.side_effect = FunctionTimedOut("timed out")

        task = TaskToBeExecuted(
            db_path="http://localhost:8890/sparql",
            queries=["SELECT ?s WHERE { ?s ?p ?o }"],
            timeout=1
        )

        results = executor.execute_queries([task])

        assert isinstance(results[0][0], ExecutorError)
        assert "Timeout" in str(results[0][0])

    @patch("NL2SQLEvaluator.db_executor_nodes.sparql_endpoint_executor.SPARQLWrapper")
    def test_parallel_execution(self, MockSPARQLWrapper, executor):
        """Multiple tasks processed concurrently."""
        mock_instance = MagicMock()
        MockSPARQLWrapper.return_value = mock_instance

        mock_instance.query.return_value.convert.return_value = {
            "results": {
                "bindings": [
                    {"val": {"type": "literal", "value": "42"}}
                ]
            }
        }

        tasks = [
            TaskToBeExecuted(
                db_path="http://localhost:8890/sparql",
                queries=[f"SELECT ?val WHERE {{ ?s <prop{i}> ?val }}"],
                timeout=30
            )
            for i in range(5)
        ]

        results = executor.execute_queries(tasks, max_workers=3)

        assert len(results) == 5
        for task_results in results:
            assert isinstance(task_results[0], SparqlExecutorOutput)
            assert task_results[0].rows == [("42",)]

    def test_read_only_allows_keywords_in_uris_and_literals(self):
        """Keywords inside URIs or string literals should not trigger the write guard."""
        # 'LOAD' inside a prefixed local name / URI
        _check_read_only("SELECT ?s WHERE { ?s a <http://example.org/Load> }")
        # 'DELETE' inside a string literal
        _check_read_only("SELECT ?s WHERE { ?s <http://ex.org/label> 'DELETE me' }")
        # 'INSERT' inside a double-quoted literal
        _check_read_only('SELECT ?s WHERE { ?s <http://ex.org/op> "INSERT" }')

    def test_write_queries_rejected_in_executor(self, executor):
        """Write queries should produce ExecutorError when run through executor."""
        task = TaskToBeExecuted(
            db_path="http://localhost:8890/sparql",
            queries=["INSERT DATA { <s> <p> <o> }"],
            timeout=30
        )

        results = executor.execute_queries([task])

        assert isinstance(results[0][0], ExecutorError)
        assert "Write operations" in str(results[0][0])
