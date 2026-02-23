"""
Integration tests for the Cypher and SPARQL evaluation pipelines.

Tests the full chain: executor (mocked) → output → evaluator → score.
No real Neo4j or SPARQL endpoint needed.
"""
from unittest.mock import MagicMock, patch

import pytest

from NL2SQLEvaluator.db_executor_nodes.db_executor_input import TaskToBeExecuted
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import (
    CypherExecutorOutput, SparqlExecutorOutput, ExecutorError,
)
from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import execute_queries_in_model_predictions
from NL2SQLEvaluator.db_executor_nodes.neo4j_db_executor import Neo4jDBExecutor
from NL2SQLEvaluator.db_executor_nodes.sparql_endpoint_executor import SparqlEndpointExecutor
from NL2SQLEvaluator.evaluator_nodes.evaluator_input import (
    EvalText2CypherTask, EvalText2SparqlTask, EvaluationType,
)
from NL2SQLEvaluator.evaluator_nodes.evaluator_protocol import evaluate_target_and_pred
from NL2SQLEvaluator.evaluator_nodes.factory_evaluator import FactoryTaskEvaluator
from NL2SQLEvaluator.evaluator_nodes.text2sql_evaluators import EXEvaluator


# ── Helpers ──────────────────────────────────────────────────────────────

def _mock_neo4j_executor(records_per_query: list[list[dict]]) -> Neo4jDBExecutor:
    """Returns a Neo4j executor with mocked driver that returns given records."""
    executor = Neo4jDBExecutor()

    call_count = {"i": 0}

    def fake_execute_read(tx_func):
        idx = call_count["i"]
        call_count["i"] += 1
        return records_per_query[idx]

    mock_session = MagicMock()
    mock_session.execute_read.side_effect = fake_execute_read

    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    return executor, mock_driver


def _mock_sparql_executor(bindings_per_query: list[list[dict]]) -> SparqlEndpointExecutor:
    """Returns a SPARQL executor with mocked SPARQLWrapper that returns given bindings."""
    executor = SparqlEndpointExecutor()
    return executor, bindings_per_query


# ── Cypher Integration ───────────────────────────────────────────────────

class TestCypherIntegration:
    """Full chain: Neo4j executor (mocked) → CypherExecutorOutput → EXEvaluator → score."""

    @patch("NL2SQLEvaluator.db_executor_nodes.neo4j_db_executor._get_or_create_driver")
    def test_executor_to_evaluator_match(self, mock_get_driver):
        """Target and prediction return same data → score 1.0."""
        # Both target and prediction return the same 2 rows
        records = [
            {"age": 30, "name": "Alice"},
            {"age": 25, "name": "Bob"},
        ]
        mock_session = MagicMock()
        mock_session.execute_read.return_value = records

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_driver.return_value = mock_driver

        executor = Neo4jDBExecutor()

        # Execute target query
        target_results = executor.execute_queries([
            TaskToBeExecuted(
                db_path="bolt://localhost:7687/testdb",
                queries=["MATCH (n:Person) RETURN n.name AS name, n.age AS age"],
                timeout=10,
            )
        ])
        # Execute prediction query (same result)
        pred_results = executor.execute_queries([
            TaskToBeExecuted(
                db_path="bolt://localhost:7687/testdb",
                queries=["MATCH (p:Person) RETURN p.name AS name, p.age AS age"],
                timeout=10,
            )
        ])

        target_output = target_results[0][0]
        pred_output = pred_results[0][0]

        assert isinstance(target_output, CypherExecutorOutput)
        assert isinstance(pred_output, CypherExecutorOutput)

        # Feed into evaluator
        evaluator = EXEvaluator()
        scores = evaluate_target_and_pred(
            evaluator,
            targets=[target_output],
            predictions=[[pred_output]],
            task_type=EvaluationType.TEXT2CYPHER,
        )

        assert scores == [1.0]

    def test_output_comparison_column_permutation(self):
        """Cypher outputs with permuted columns should still be equivalent."""
        target = CypherExecutorOutput(rows=[(30, "Alice"), (25, "Bob")])
        pred = CypherExecutorOutput(rows=[("Alice", 30), ("Bob", 25)])

        assert target.is_equivalent_to(pred, is_row_order_important=False)

    def test_output_comparison_mismatch(self):
        """Different data → not equivalent."""
        target = CypherExecutorOutput(rows=[(30, "Alice")])
        pred = CypherExecutorOutput(rows=[(99, "Charlie")])

        assert not target.is_equivalent_to(pred, is_row_order_important=False)

    def test_evaluator_with_error_prediction(self):
        """If prediction is an ExecutorError, score should be 0.0."""
        target = CypherExecutorOutput(rows=[(1,)])
        pred = CypherExecutorOutput(rows=[])  # empty result = mismatch

        evaluator = EXEvaluator()
        scores = evaluate_target_and_pred(
            evaluator,
            targets=[target],
            predictions=[[pred]],
            task_type=EvaluationType.TEXT2CYPHER,
        )

        assert scores == [0.0]

    def test_factory_creates_correct_task_type(self):
        """FactoryTaskEvaluator routes text2cypher to EvalText2CypherTask."""
        target = CypherExecutorOutput(rows=[(1,)])
        preds = [CypherExecutorOutput(rows=[(1,)])]

        task = FactoryTaskEvaluator.create(
            task_type=EvaluationType.TEXT2CYPHER,
            target=target,
            predictions=preds,
        )
        assert isinstance(task, EvalText2CypherTask)

    def test_majority_voting(self):
        """With 3 predictions (2 agree), majority vote picks the majority."""
        target = CypherExecutorOutput(rows=[(42,)])
        pred_a = CypherExecutorOutput(rows=[(42,)])
        pred_b = CypherExecutorOutput(rows=[(99,)])

        task = EvalText2CypherTask(
            target=target,
            predictions=[pred_a, pred_a, pred_b],
        )
        best = task.get_best_pred_by_majority_voting(is_row_order_important=False)
        assert best.rows == [(42,)]


# ── SPARQL Integration ───────────────────────────────────────────────────

class TestSparqlIntegration:
    """Full chain: SPARQL executor (mocked) → SparqlExecutorOutput → EXEvaluator → score."""

    @patch("NL2SQLEvaluator.db_executor_nodes.sparql_endpoint_executor.func_timeout")
    @patch("NL2SQLEvaluator.db_executor_nodes.sparql_endpoint_executor.SPARQLWrapper")
    def test_executor_to_evaluator_match(self, MockSPARQLWrapper, mock_func_timeout):
        """Target and prediction return same data → score 1.0."""
        bindings = [
            {"name": {"type": "literal", "value": "Alice"}, "age": {"type": "literal", "value": "30"}},
            {"name": {"type": "literal", "value": "Bob"}, "age": {"type": "literal", "value": "25"}},
        ]

        mock_instance = MagicMock()
        MockSPARQLWrapper.return_value = mock_instance
        mock_instance.query.return_value.convert.return_value = {
            "results": {"bindings": bindings}
        }

        # Make func_timeout call the function directly
        mock_func_timeout.side_effect = lambda timeout, fn: fn()

        executor = SparqlEndpointExecutor()

        target_results = executor.execute_queries([
            TaskToBeExecuted(
                db_path="http://localhost:8890/sparql",
                queries=["SELECT ?name ?age WHERE { ?s foaf:name ?name ; foaf:age ?age }"],
                timeout=30,
            )
        ])
        pred_results = executor.execute_queries([
            TaskToBeExecuted(
                db_path="http://localhost:8890/sparql",
                queries=["SELECT ?age ?name WHERE { ?s foaf:name ?name ; foaf:age ?age }"],
                timeout=30,
            )
        ])

        target_output = target_results[0][0]
        pred_output = pred_results[0][0]

        assert isinstance(target_output, SparqlExecutorOutput)
        assert isinstance(pred_output, SparqlExecutorOutput)

        # Feed into evaluator
        evaluator = EXEvaluator()
        scores = evaluate_target_and_pred(
            evaluator,
            targets=[target_output],
            predictions=[[pred_output]],
            task_type=EvaluationType.TEXT2SPARQL,
        )

        assert scores == [1.0]

    def test_output_comparison_column_permutation(self):
        """SPARQL outputs with permuted columns should still be equivalent."""
        target = SparqlExecutorOutput(rows=[("30", "Alice"), ("25", "Bob")])
        pred = SparqlExecutorOutput(rows=[("Alice", "30"), ("Bob", "25")])

        assert target.is_equivalent_to(pred, is_row_order_important=False)

    def test_output_comparison_mismatch(self):
        target = SparqlExecutorOutput(rows=[("Alice",)])
        pred = SparqlExecutorOutput(rows=[("Charlie",)])

        assert not target.is_equivalent_to(pred)

    def test_factory_creates_correct_task_type(self):
        """FactoryTaskEvaluator routes text2sparql to EvalText2SparqlTask."""
        target = SparqlExecutorOutput(rows=[(1,)])
        preds = [SparqlExecutorOutput(rows=[(1,)])]

        task = FactoryTaskEvaluator.create(
            task_type=EvaluationType.TEXT2SPARQL,
            target=target,
            predictions=preds,
        )
        assert isinstance(task, EvalText2SparqlTask)

    def test_majority_voting(self):
        """With 3 predictions (2 agree), majority vote picks the majority."""
        target = SparqlExecutorOutput(rows=[("42",)])
        pred_a = SparqlExecutorOutput(rows=[("42",)])
        pred_b = SparqlExecutorOutput(rows=[("99",)])

        task = EvalText2SparqlTask(
            target=target,
            predictions=[pred_a, pred_a, pred_b],
        )
        best = task.get_best_pred_by_majority_voting(is_row_order_important=False)
        assert best.rows == [("42",)]


# ── Cross-cutting: db_executor_protocol grouping ─────────────────────────

class TestExecutorProtocolGrouping:
    """Test that execute_queries_in_model_predictions works with graph executors."""

    @patch("NL2SQLEvaluator.db_executor_nodes.neo4j_db_executor._get_or_create_driver")
    def test_cypher_grouping_by_db_path(self, mock_get_driver):
        """Queries to the same bolt URI are grouped into one task."""
        mock_session = MagicMock()
        mock_session.execute_read.return_value = [{"val": 1}]

        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_driver.return_value = mock_driver

        executor = Neo4jDBExecutor()

        results = execute_queries_in_model_predictions(
            code_executor=executor,
            db_paths=[
                "bolt://localhost:7687/db1",
                "bolt://localhost:7687/db1",
                "bolt://localhost:7687/db2",
            ],
            queries=[
                ["RETURN 1 AS val"],
                ["RETURN 1 AS val"],
                ["RETURN 1 AS val"],
            ],
            timeout=10,
        )

        assert len(results) == 3
        for batch in results:
            assert len(batch) == 1
            assert isinstance(batch[0], CypherExecutorOutput)
