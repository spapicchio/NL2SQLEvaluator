import math
from typing import List, Tuple

import pytest

from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SQLExecutorOutput
from NL2SQLEvaluator.evaluator_nodes.evaluator_input import EvalText2SQLTask
from NL2SQLEvaluator.evaluator_nodes.text2sql_evaluators import BirdEXEvaluator


@pytest.fixture
def bird_executor() -> BirdEXEvaluator:
    """Fixture to provide a clean BirdEXEvaluator instance for each test.

    Why: Isolates evaluator state between tests.
    """
    return BirdEXEvaluator()


class TestBirdEX:
    def _run_eval(self, executor: BirdEXEvaluator, tar_rows: List[Tuple], pred_rows: List[Tuple]) -> float:
        """Helper to wrap rows into a Task and execute the metric.

        Args:
            executor: The evaluator instance.
            tar_rows: Ground truth rows.
            pred_rows: Model predicted rows.

        Returns:
            The score (usually 0.0 or 1.0).
        """
        task = EvalText2SQLTask(
            target=SQLExecutorOutput(rows=tar_rows),
            predictions=[SQLExecutorOutput(rows=pred_rows)]
        )
        # Assuming execute_metric returns a list of floats for the provided tasks
        results = executor.execute_metric(tasks=[task])
        return results[0]

    @pytest.mark.parametrize("test_id, tar, pred, expected", [
        # Base evaluations:
        ("exact_match", [('a', 'b'), ('c', 'd')], [('a', 'b'), ('c', 'd')], 1.0),
        ("different_values", [('a', 'b')], [('c', 'd')], 0.0),
        ("null_equivalence", [('a', None)], [('a', None)], 1.0),
        ("nan_equivalence", [('a', math.nan)], [('a', math.nan)], 1.0),
        ("floating_point_precision", [('a', 1)], [('a', 1.0000000001)], 0.0),
        ("empty_vs_non_empty", [], [('a',)], 0.0),
        ("both_empty", [], [], 1.0),
        ("nan_vs_string_nan", [(math.nan,)], [('NaN',)], 0.0),
        # BIRD-specific logic:
        # order by is not calculated in BirdEX, so this should still be a match
        ("reordered_rows", [('a', 'b'), ('c', 'd')], [('c', 'd'), ('a', 'b')], 1.0),
        # The metric is not column invariant, so this should fail
        ("different_column_projection", [('a', 'b')], [('b', 'a')], 0.0),
        # BirdEX is not a multiset metric, so duplicate rows in the target should not cause a mismatch
        ("multiset_collapse_bird_ex", [('a', 'b'), ('a', 'b')], [('a', 'b')], 1.0),
    ])
    def test_bird_ex_logic(self, bird_executor, test_id, tar, pred, expected):
        """Validates various row comparison scenarios.

        The 'multiset_collapse' test specifically checks BirdEX's behavior
        regarding duplicate rows in the target vs prediction.
        """
        score = self._run_eval(bird_executor, tar, pred)
        assert score == expected, f"Failed case '{test_id}': expected {expected}, got {score}"

    def test_bulk_processing(self, bird_executor):
        """Verifies that the evaluator handles multiple tasks in one call.

        Why: Ensures batch processing doesn't leak state between tasks.
        """
        tasks = [
            EvalText2SQLTask(target=SQLExecutorOutput(rows=[(1,)]), predictions=[SQLExecutorOutput(rows=[(1,)])]),
            EvalText2SQLTask(target=SQLExecutorOutput(rows=[(1,)]), predictions=[SQLExecutorOutput(rows=[(2,)])]),
            EvalText2SQLTask(target=SQLExecutorOutput(rows=[(1,)]), predictions=[SQLExecutorOutput(rows=[(2,)])]),
            EvalText2SQLTask(target=SQLExecutorOutput(rows=[(1,)]), predictions=[SQLExecutorOutput(rows=[(2,)])]),
        ]
        results = bird_executor.execute_metric(tasks=tasks)
        assert results == [1.0, 0.0, 0.0, 0.0], f"Expected [1.0, 0.0, 0.0, 0.0], got {results}"
