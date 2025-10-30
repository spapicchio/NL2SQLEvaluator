import pytest

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import OutputTable
from NL2SQLEvaluator.evaluator_nodes import BirdEXEvaluator
from NL2SQLEvaluator.evaluator_nodes.evaluator_protocol import EvaluateTask


@pytest.fixture
def executor() -> BirdEXEvaluator:
    return BirdEXEvaluator()


class TestBirdEX:
    def _internal_run(self, tar, pred, executor):
        task = EvaluateTask(
            predictions=[OutputTable(rows=pred[0][0])],
            target=[OutputTable(rows=tar[0][0])]
        )

        result = executor.execute_metric(
            tasks=[task]
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
        assert result[0] == 0.0

    def test_equal_but_different_projection(self, executor):
        multiple_tasks_tar = [('a', 'b'), ('c', 'd')]
        multiple_tasks_preds = [('b', 'a'), ('d', 'c')]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 0.0

    def test_equal_but_different_tuple_order(self, executor):
        multiple_tasks_tar = [('a', 'b'), ('c', 'd')]
        multiple_tasks_preds = [('c', 'd'), ('a', 'b')]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 1.0

    def test_null_values(self, executor):
        multiple_tasks_tar = [('a', None), ('c', 'd')]
        multiple_tasks_preds = [('c', 'd'), ('a', None)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 1.0

    def test_null_math_none(self, executor):
        import math
        multiple_tasks_tar = [('a', math.nan), ('c', 'd')]
        multiple_tasks_preds = [('c', 'd'), ('a', math.nan)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 1.0

    def test_mixed_types(self, executor):
        multiple_tasks_tar = [('a', 1), ('c', 'd')]
        multiple_tasks_preds = [('c', 'd'), ('a', 1.0000000001)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 0.0

    def test_empty(self, executor):
        multiple_tasks_tar = []
        multiple_tasks_preds = [('c', 'd'), ('a', 1.0000000001)]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 0.0
        multiple_tasks_tar = []
        multiple_tasks_preds = []
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
        assert result[0] == 0.0

    def test_bird_ex_no_distinct(self, executor):
        multiple_tasks_tar = [('a', 'b'), ('a', 'b')]
        multiple_tasks_preds = [('a', 'b')]
        result = self._internal_run([[multiple_tasks_tar]], [[multiple_tasks_preds]], executor)
        assert result[0] == 1.0
