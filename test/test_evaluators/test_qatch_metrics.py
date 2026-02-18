from typing import List, Tuple

import pytest

from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SQLExecutorOutput
from NL2SQLEvaluator.evaluator_nodes.evaluator_input import EvalText2SQLTask
from NL2SQLEvaluator.evaluator_nodes.text2sql_evaluators.qatch_metrics import QATCHEvaluator, QATCHMetric


@pytest.fixture
def qatch_executor() -> QATCHEvaluator:
    """Fixture for the QATCH evaluator instance."""
    return QATCHEvaluator()


class BaseTestQATCH:
    """Base logic for QATCH metric testing to avoid code duplication."""

    def _run_metric(self, executor: QATCHEvaluator, metric: QATCHMetric,
                    tar: List[Tuple], pred: List[Tuple]) -> float:
        """Standard wrapper to execute a QATCH metric."""
        task = EvalText2SQLTask(
            predictions=[SQLExecutorOutput(rows=pred)],
            target=SQLExecutorOutput(rows=tar)
        )
        result = executor.execute_metric(tasks=[task], metric=metric)
        return result[0]


class TestQATCHMetrics(BaseTestQATCH):
    """Combined test suite for Tuple Cardinality, Cell Precision, and Cell Recall."""

    @pytest.mark.parametrize("metric", [
        QATCHMetric.TUPLE_CARDINALITY,
        QATCHMetric.CELL_PRECISION,
        QATCHMetric.CELL_RECALL
    ])
    @pytest.mark.parametrize("test_id, tar, pred, expected_map", [
        ("exact_match", [('a', 'b')], [('a', 'b')], 1.0),
        ("diff_order", [('a', 'b'), ('c', 'd')], [('c', 'd'), ('a', 'b')], 1.0),
        ("diff_proj", [('a', 'b')], [('b', 'a')], 1.0),
        ("null_match", [('a', None)], [('a', None)], 1.0),
        ("both_empty", [], [], 1.0),
        ("empty_vs_val", [], [('a',)], 0.0),
        # Metric specific expectations: {Metric: Score}
        (
                "subset",
                [('a', 'b'), ('c', 'd')],
                [('c', 'd')],
                {
                    QATCHMetric.TUPLE_CARDINALITY: 0.5,
                    QATCHMetric.CELL_PRECISION: 1.0,
                    QATCHMetric.CELL_RECALL: 0.5}
        ),
    ])
    def test_shared_scenarios(self, qatch_executor, metric, test_id, tar, pred, expected_map):
        """Tests standard SQL result scenarios across multiple metrics."""
        # Handle cases where expected score is the same for all or unique to metric
        expected = expected_map[metric] if isinstance(expected_map, dict) else expected_map

        score = self._run_metric(qatch_executor, metric, tar, pred)
        assert score == pytest.approx(expected), f"Metric {metric} failed on {test_id}"

    @pytest.mark.parametrize("metric, tar, pred, expected", [
        # Cell Recall special cases
        (QATCHMetric.CELL_RECALL, [('A', 'B', 'C')], [('A',)], 1 / 3),
        # Tuple Order special cases
        (QATCHMetric.TUPLE_ORDER, [('a',), ('b',)], [('b',), ('a',)], 0.0),
        (QATCHMetric.TUPLE_ORDER, [('a',), ('b',)], [('a',), ('b',)], 1.0),
    ])
    def test_specialized_cases(self, qatch_executor, metric, tar, pred, expected):
        """Tests logic unique to specific metrics (e.g. Order or partial Recall)."""
        score = self._run_metric(qatch_executor, metric, tar, pred)
        assert score == pytest.approx(expected, abs=1e-3)

    def test_mixed_types_precision_recall(self, qatch_executor):
        """Specific check for precision/recall behavior on mixed types/NaN."""
        tar = [('a', 1)]
        pred = [('a', 1.0000000001)]  # Mismatch

        # Precision checks accuracy of cells found
        p_score = self._run_metric(qatch_executor, QATCHMetric.CELL_PRECISION, tar, pred)
        assert p_score == 0.5  # 'a' matches, 1 != 1.0000000001

    def test_tuple_constraint_duplicates(self, qatch_executor):
        """Tests Tuple Constraint logic regarding duplicates."""
        metric = QATCHMetric.TUPLE_CONSTRAINT
        tar = [('a', 'b'), ('a', 'b'), ('c', 'd')]
        pred = [('b', 'a'), ('c', 'd')]

        score = self._run_metric(qatch_executor, metric, tar, pred)
        # Assuming TupleConstraint penalizes missing duplicates
        assert score == 0.5
