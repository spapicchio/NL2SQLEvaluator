import math
import pytest
from NL2SQLEvaluator.metric_executor.evaluator_orchestrator import evaluator_orchestrator, OrchestratorInput

class TestExecutionAccuracy:
    @pytest.fixture
    def orchestrator_input(self):
        return {
            "executor": None,
            "metrics_to_calculate": ["execution_accuracy"]
        }

    def test_equal(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['execution_accuracy'] == 1.0

    def test_different(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['execution_accuracy'] == 0.0

    def test_equal_but_different_projection(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('b', 'a'), ('d', 'c')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        # unfortunately, in standard execution accuracy, projection order does not matter :(
        assert result[0]['execution_accuracy'] == 0.0

    def test_equal_but_different_tuple_order(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('c', 'd'), ('a', 'b')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['execution_accuracy'] == 1.0

    def test_null_values(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', None), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('c', 'd'), ('a', None)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['execution_accuracy'] == 1.0

    def test_null_math_none(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', math.nan), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('c', 'd'), ('a', math.nan)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['execution_accuracy'] == 1.0

    def test_mixed_types(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 1), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('c', 'd'), ('a', 1.0000000001)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['execution_accuracy'] == 1.0

    def test_empty_lists(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['execution_accuracy'] == 1.0

    def test_empty_vs_non_empty(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[('a',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['execution_accuracy'] == 0.0

    def test_nan_and_string_mismatch(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[(math.nan,)]]
        orchestrator_input["predicted_queries"] = [[('NaN',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['execution_accuracy'] == 0.0

    def test_float_outside_epsilon(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[(1.0,)]]
        orchestrator_input["predicted_queries"] = [[(1.00001,)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['execution_accuracy'] == 0.0  # assuming default epsilon=1e-6

    def test_different_tuple_lengths(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b')]]
        orchestrator_input["predicted_queries"] = [[('a',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['execution_accuracy'] == 0.0

    def test_bird_ex_no_distinct(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('a', 'b')]]
        orchestrator_input["predicted_queries"] = [[('a', 'b')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        # unfortunately, in standard execution accuracy, duplicates do not matter :(
        assert result[0]['execution_accuracy'] == 1.0