import pytest
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SQLExecutorOutput
from NL2SQLEvaluator.evaluator_nodes.text2sql_evaluators.ambig_text2sql_evaluators import AmbigText2SQLEvaluator
from NL2SQLEvaluator.evaluator_nodes.evaluator_input import EvalAmbigText2SQLTask


@pytest.fixture
def ambig_evaluator() -> AmbigText2SQLEvaluator:
    return AmbigText2SQLEvaluator()


class TestAmbigText2SQLEvaluatorEdgeCases:
    """Focuses on the deduplication and threshold logic of the ambiguous evaluator."""

    def _make_task(self, targets: list[list[tuple]], preds: list[list[tuple]]) -> EvalAmbigText2SQLTask:
        return EvalAmbigText2SQLTask(
            target=[SQLExecutorOutput(rows=r) for r in targets],
            predictions=[SQLExecutorOutput(rows=r) for r in preds]
        )

    def test_permutation_deduplication(self, ambig_evaluator):
        """Verify that different row orders are treated as the same unique prediction when order is not important."""
        # Two predictions that are just permutations of each other
        pred_rows = [[(1,), (2,)], [(2,), (1,)]]
        target_rows = [[(1,), (2,)]]

        task = self._make_task(target_rows, pred_rows)

        # When order is NOT important, they collapse to 1 unique prediction.
        # Since that prediction is correct, precision = 1/1 = 1.0
        res = ambig_evaluator.execute_metric(
            tasks=[task],
            metric='precision',
            is_row_order_important=False
        )
        assert res[0] == 1.0

    def test_threshold_boundary(self, ambig_evaluator):
        """Verify the 0.6 match threshold."""
        # We need a case where the score is slightly below 0.6.
        # Using QATCH: (CellP + CellR + Card) / 3
        # Target: (1, 2, 3) | Pred: (1, 2)
        # CP: 1.0, CR: 0.66, Card: 0.66 -> Avg: 0.77 (Matches)
        # Target: (1, 2, 3, 4, 5) | Pred: (1, 2)
        # CP: 1.0, CR: 0.4, Card: 0.4 -> Avg: 0.6 (Exactly matches threshold)

        target = [[(1,), (2,), (3,), (4,), (5,)]]
        pred = [[(1,), (2,)]]  # Score 0.6

        task = self._make_task(target, pred)
        res = ambig_evaluator.execute_metric(
            tasks=[task],
            metric='precision',
            evaluate_with='qatch',
            threshold=0.6
        )
        assert res[0] == 1.0

    def test_many_to_many_matching(self, ambig_evaluator):
        """Verify that multiple targets can be satisfied by multiple distinct predictions."""
        # Model provides two different interpretations, both are valid
        targets = [[(1,)], [(2,)]]
        preds = [[(1,)], [(2,)]]

        task = self._make_task(targets, preds)

        # Recall: Both targets hit? (2/2)
        recall = ambig_evaluator.execute_metric(tasks=[task], metric='recall')[0]
        # Precision: Both predictions correct? (2/2)
        precision = ambig_evaluator.execute_metric(tasks=[task], metric='precision')[0]

        assert recall == 1.0
        assert precision == 1.0

    def test_duplicate_targets_in_task(self, ambig_evaluator):
        """Verify that identical targets in ground truth are deduplicated."""
        # Ground truth accidentally includes the same interpretation twice
        targets = [[(1,)], [(1,)]]
        preds = [[(1,)]]

        task = self._make_task(targets, preds)

        # If target deduplication works, len(target2match) is 1.
        # 1 match / 1 unique target = 1.0 recall
        res = ambig_evaluator.execute_metric(tasks=[task], metric='recall')
        assert res[0] == 1.0

    def test_bulk_task_processing(self, ambig_evaluator):
        """Verify the evaluator handles a list of multiple tasks correctly."""
        tasks = [
            self._make_task([[(1,)]], [[(1,)]]),  # 1.0
            self._make_task([[(1,)]], [[(2,)]])  # 0.0
        ]
        results = ambig_evaluator.execute_metric(tasks=tasks, metric='precision')
        assert results == [1.0, 0.0]

    def test_is_row_order_important_consistency(self, ambig_evaluator):
        """Check that the flag is correctly passed down to the underlying evaluators."""
        # Row order matters for EX
        targets = [[(1,), (2,)]]
        preds = [[(2,), (1,)]]

        task = self._make_task(targets, preds)

        # Should fail with row order
        res_fail = ambig_evaluator.execute_metric(
            tasks=[task], metric='precision', is_row_order_important=True
        )
        assert res_fail[0] == 0.0

        # Should pass without row order
        res_pass = ambig_evaluator.execute_metric(
            tasks=[task], metric='precision', is_row_order_important=False
        )
        assert res_pass[0] == 1.0