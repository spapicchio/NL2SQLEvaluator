"""Unit tests for the NL2SQL Evaluation Task module."""

import pytest

from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SQLExecutorOutput
from NL2SQLEvaluator.evaluator_nodes.evaluator_input import EvaluationType, EvalText2SQLTask
from NL2SQLEvaluator.evaluator_nodes.factory_evaluator import FactoryTaskEvaluator


# from your_module import EvaluationTaskFactory, EvaluationType, EvalText2SQLTask

class TestEvaluationTaskFactory:
    """Tests the factory's ability to instantiate the correct task types.

    Why: Ensures the mapping between EvaluationType and Class is maintained.
    How: Run `pytest` to validate factory creation logic.
    """

    @pytest.fixture()
    def list_sql_output(self):
        return [SQLExecutorOutput(rows=[(1,), ('A',)])]

    def test_create_text2sql(self, list_sql_output):
        task = FactoryTaskEvaluator.create(
            task_type=EvaluationType.TEXT2SQL,
            predictions=list_sql_output,
            target=list_sql_output[0]
        )
        assert isinstance(task, EvalText2SQLTask)
        assert task.task_type == EvaluationType.TEXT2SQL

    def test_create_invalid_type(self):
        with pytest.raises(ValueError):
            FactoryTaskEvaluator.create("invalid", [], [])  # pyrefly: ignore


class TestEvalText2SQLTask:
    """Tests the specific logic for standard Text2SQL tasks.

    Why: Majority voting is the core selection mechanism for Self-Consistency.
    """

    def test_majority_voting_with_clear_winner(self):
        # Setup: 3 predictions, two have identical rows [2, 2]
        p_wrong = SQLExecutorOutput(rows=[(1,)])
        p_right_1 = SQLExecutorOutput(rows=[(1, 'A'), (2, 2)])
        p_right_2 = SQLExecutorOutput(rows=[(2, 2), ('A', 1)])

        task = EvalText2SQLTask(
            target=p_right_1,
            predictions=[p_wrong, p_right_1, p_right_2]
        )

        assert task.get_best_pred_by_majority_voting(is_row_order_important=False) == p_right_1
        assert task.get_best_pred_by_majority_voting(is_row_order_important=True) == p_wrong

    def test_majority_voting_with_tie(self):
        # Setup: 3 predictions, two have identical rows [2, 2]
        p_wrong = SQLExecutorOutput(rows=[(1,)])
        p_right_1 = SQLExecutorOutput(rows=[(1, 'A'), (2, 2)])

        task = EvalText2SQLTask(
            target=p_right_1,
            predictions=[p_wrong, p_right_1]
        )
        assert task.get_best_pred_by_majority_voting(is_row_order_important=True) == p_wrong

    def test_majority_voting_row_order_insensitivity(self):
        # Setup: Rows are same but different order
        # Setup: 3 predictions, two have identical rows [2, 2]
        p_wrong = SQLExecutorOutput(rows=[(1,)])
        p_right_1 = SQLExecutorOutput(rows=[(1, 'A'), (2, 2)])
        p_right_2 = SQLExecutorOutput(rows=[(2, 2), ('A', 1)])

        task = EvalText2SQLTask(
            target=p_right_1,
            predictions=[p_wrong, p_right_1, p_right_2]
        )

        assert task.get_best_pred_by_majority_voting(is_row_order_important=False) == p_right_1
        assert task.get_best_pred_by_majority_voting(is_row_order_important=True) == p_wrong
