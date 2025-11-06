import pytest

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import OutputTable
from NL2SQLEvaluator.evaluator_nodes.evaluator_protocol import EvaluateTask
from NL2SQLEvaluator.evaluator_nodes.qatch_metrics import QATCHEvaluator


@pytest.fixture
def executor() -> QATCHEvaluator:
    return QATCHEvaluator()


class TestTupleOrder:
    def _internal_run(self, tar, pred, executor):
        task = EvaluateTask(
            predictions=[OutputTable(rows=pred[0][0])],
            target=[OutputTable(rows=tar[0][0])]
        )

        result = executor.execute_metric(
            tasks=[task],
            metric='tuple_order'
        )
        return result

    def test_equal(self, executor):
        multiple_tasks_tar = [('a', 'b'), ('c', 'd')]
        multiple_tasks_preds = [('a', 'b'), ('c', 'd')]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 1.0

    def test_different(self, executor):
        multiple_tasks_tar = [('a', 'b'), ('c', 'd')]
        multiple_tasks_preds = [('c', 'd')]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 1.0

    def test_empty(self, executor):
        multiple_tasks_tar = []
        multiple_tasks_preds = [('c', 'd'), ('a', 1.0000000001)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 0.0
        multiple_tasks_tar = []
        multiple_tasks_preds = []
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 1.0

    def test_equal_but_different_projection(self, executor):
        multiple_tasks_tar = [('a', 'b'), ('c', 'd')]
        multiple_tasks_preds = [('b', 'a'), ('d', 'c')]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 1.0

    def test_equal_but_different_tuple_order(self, executor):
        multiple_tasks_tar = [('a', 'b'), ('c', 'd')]
        multiple_tasks_preds = [('c', 'd'), ('a', 'b')]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 0.0

        multiple_tasks_tar = [('a', 'b'), ('c', 'd')]
        multiple_tasks_preds = [('b', 'a'), ('c', 'd')]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 1.0

    def test_null_values(self, executor):
        multiple_tasks_tar = [('a', None), ('c', 'd')]
        multiple_tasks_preds = [('c', 'd'), ('a', None)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 0.0

    def test_null_math_none(self, executor):
        import math
        multiple_tasks_tar = [('a', math.nan), ('c', 'd')]
        multiple_tasks_preds = [('a', math.nan), ('c', 'd')]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 1.0

    def test_mixed_types(self, executor):
        multiple_tasks_tar = [('a', 1), ('c', 'd')]
        multiple_tasks_preds = [('c', 'd'), ('a', 1.0000000001)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 1.0

    def test_empty_lists(self, executor):
        multiple_tasks_tar = []
        multiple_tasks_preds = []
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 1.0

    def test_empty_vs_non_empty(self, executor):
        multiple_tasks_tar = []
        multiple_tasks_preds = [('a',)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 0.0

    def test_nan_and_string_mismatch(self, executor):
        import math
        multiple_tasks_tar = [(math.nan,)]
        multiple_tasks_preds = [('NaN',)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 0.5

    def test_bird_ex_no_distinct(self, executor):
        multiple_tasks_tar = [('a', 'b')]
        multiple_tasks_preds = [('a',)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 0.5

    def test_float_outside_epsilon(self, executor):
        multiple_tasks_tar = [(1.0,)]
        multiple_tasks_pred = [(1.00001,)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_pred]], executor)
        assert result[0] == 0.5  # assuming default epsilon=1e-6

    def test_different_tuple_lengths(self, executor):
        multiple_tasks_tar = [('a', 'b')]
        multiple_tasks_pred = [('a',)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_pred]], executor)
        assert result[0] == 0.5

    def test_evaluate_opposite_direction(self, executor):
        multiple_tasks_tar = [('apple', 'orange'), ('pear',)]
        multiple_tasks_pred = [('pear',), ('apple', 'orange')]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_pred]], executor)
        assert result[0] == 0.0

        multiple_tasks_tar = [('a',), ('b',), (None,)]
        multiple_tasks_pred = [(None,), ('b',), ('a',)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_pred]], executor)
        assert result[0] == 0.0

    def test_evaluate_empty_input(self, executor):
        multiple_tasks_tar = [[]]
        multiple_tasks_pred = [[]]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_pred]], executor)
        assert result[0] == 1.0

    def test_evaluate_same_order(self, executor):
        multiple_tasks_tar = [('a',), ('b',), ('c',)]
        multiple_tasks_pred = [('a',), ('b',), ('c',)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_pred]], executor)
        assert result[0] == 1.0

    def test_evaluate_more_elements_pred(self, executor):
        multiple_tasks_tar = [('a',), ('b',), ('c',), ('d',)]
        multiple_tasks_pred = [('c',), ('b',), ('e',), ('f',)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_pred]], executor)
        # prediction after normalization  = [['c'], ['b']]
        assert result[0] == 0.0

    def test_evaluate_more_elements_target(self, executor):
        multiple_tasks_tar = [('a',), ('a',), ('a',), ('b',), ('c',), ('d',)]
        multiple_tasks_pred = [('a',), ('b',), ('b',), ('c',), ('e',)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_pred]], executor)
        assert result[0] == 1.0

    def test_evaluate_no_correlation(self, executor):
        multiple_tasks_tar = [('a',), ('b',), ('c',)]
        multiple_tasks_pred = [('d',), ('e',), ('f',)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_pred]], executor)
        score = result[0]
        assert score != 0.0 and score != 1.0
