import math
import pytest
from NL2SQLEvaluator.metric_executor.evaluator_orchestrator import evaluator_orchestrator, OrchestratorInput

class TestTupleCardinality:
    @pytest.fixture
    def orchestrator_input(self):
        return {
            "executor": None,
            "metrics_to_calculate": ["tuple_cardinality"]
        }

    def test_equal(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == 1.0

    def test_different(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == 0.5

    def test_equal_but_different_projection(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('b', 'a'), ('d', 'c')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == 1.0

    def test_equal_but_different_tuple_order(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('d', 'c'), ('b', 'a')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == 1.0

    def test_null_values(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', None), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('d', 'c'), (None, 'a')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == 1.0

    def test_null_math_none(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', math.nan), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('d', 'c'), (math.nan, 'a')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == 1.0

    def test_mixed_types(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 1), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('d', 'c'), (1.0000000001, 'a')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == 1.0

    def test_empty_lists(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == 1.0

    def test_empty_vs_non_empty(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[('a',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == 0.0

    def test_nan_and_string_mismatch(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[(math.nan,)]]
        orchestrator_input["predicted_queries"] = [[('NaN',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == 1.0

    def test_float_outside_epsilon(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[(1.0,)]]
        orchestrator_input["predicted_queries"] = [[(1.00001,)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == 1.0  # assuming default epsilon=1e-6

    def test_different_tuple_lengths(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b')]]
        orchestrator_input["predicted_queries"] = [[('a',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == 1.0

    def test_target_greater_than_prediction(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd'), ('e', 'f')]]
        orchestrator_input["predicted_queries"] = [[('a', 'b')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == round(1 / 3, 3)

    def test_prediction_greater_than_target(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b')]]
        orchestrator_input["predicted_queries"] = [[('a', 'b'), ('c', 'd'), ('e', 'f')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == round(1 / 3, 3)

    def test_equal_target_and_prediction(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == 1.0

    def test_empty_target_and_prediction(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == 1.0

    def test_empty_prediction_with_non_empty_target(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == 0.0

    def test_non_empty_prediction_with_empty_target(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_cardinality'] == 0.0