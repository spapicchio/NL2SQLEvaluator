import math

import pytest

from NL2SQLEvaluator.metric_executor.evaluator_orchestrator import evaluator_orchestrator, OrchestratorInput


class TestTupleOrder:
    @pytest.fixture
    def orchestrator_input(self):
        return {
            "executor": None,
            "metrics_to_calculate": ["tuple_order"]
        }

    def test_equal(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_order'] == 1.0

    def test_different(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_order'] == 1.0

    def test_equal_but_different_projection(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('b', 'a'), ('d', 'c')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_order'] == 1.0

    def test_equal_but_different_tuple_order(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('d', 'c'), ('b', 'a')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_order'] == 0.0

    def test_null_values(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', None), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('d', 'c'), (None, 'a')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_order'] == 0.0

    def test_null_math_none(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', math.nan), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('d', 'c'), ('a', math.nan)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_order'] == 0.0

    def test_mixed_types(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 1), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[(1.0000000001, 'a'), ('d', 'c')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_order'] == 1.0

    def test_empty_lists(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_order'] == 1.0

    def test_empty_vs_non_empty(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[('a',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_order'] == 0.0

    def test_nan_and_string_mismatch(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[(math.nan,)]]
        orchestrator_input["predicted_queries"] = [[('NaN',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_order'] == 0.5

    def test_float_outside_epsilon(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[(1.0,)]]
        orchestrator_input["predicted_queries"] = [[(1.00001,)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_order'] == 0.5  # assuming default epsilon=1e-6

    def test_different_tuple_lengths(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b')]]
        orchestrator_input["predicted_queries"] = [[('a',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_order'] == 0.5

    def test_evaluate_opposite_direction(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('apple', 'orange'), ('pear',)]]
        orchestrator_input["predicted_queries"] = [[('pear',), ('apple', 'orange')]]
        result = evaluator_orchestrator.invoke(OrchestratorInput(**orchestrator_input))
        assert result[0]['tuple_order'] == 0.0

        orchestrator_input["target_queries"] = [[('a',), ('b',), (None,)]]
        orchestrator_input["predicted_queries"] = [[(None,), ('b',), ('a',)]]
        result = evaluator_orchestrator.invoke(OrchestratorInput(**orchestrator_input))
        assert result[0]['tuple_order'] == 0.0

    def test_evaluate_empty_input(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_order'] == 1.0

    def test_evaluate_same_order(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a',), ('b',), ('c',)]]
        orchestrator_input["predicted_queries"] = [[('a',), ('b',), ('c',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_order'] == 1.0

    def test_evaluate_more_elements_pred(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a',), ('b',), ('c',), ('d',)]]
        orchestrator_input["predicted_queries"] = [[('c',), ('b',), ('e',), ('f',)]]
        # prediction after normalization  = [['c'], ['b']]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_order'] == 0.0

    def test_evaluate_more_elements_target(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a',), ('a',), ('a',), ('b',), ('c',), ('d',)]]
        orchestrator_input["predicted_queries"] = [[('a',), ('b',), ('b',), ('c',), ('e',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_order'] == 1.0

    def test_evaluate_no_correlation(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a',), ('b',), ('c',)]]
        orchestrator_input["predicted_queries"] = [[('d',), ('e',), ('f',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        score = result[0]['tuple_order']
        assert score != 0.0 and score != 1.0
