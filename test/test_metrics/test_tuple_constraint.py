import math
import pytest
from NL2SQLEvaluator.metric_executor.evaluator_orchestrator import evaluator_orchestrator, OrchestratorInput

class TestTupleConstraint:
    @pytest.fixture
    def orchestrator_input(self):
        return {
            "executor": None,
            "metrics_to_calculate": ["tuple_constraint"]
        }

    def test_equal(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 1.0

    def test_different(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 0.5

    def test_equal_but_different_projection(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('b', 'a'), ('d', 'c')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 1.0

    def test_equal_but_different_tuple_order(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('d', 'c'), ('b', 'a')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 1.0

    def test_null_values(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', None), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('d', 'c'), (None, 'a')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 1.0

    def test_null_math_none(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', math.nan), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('d', 'c'), (math.nan, 'a')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 1.0

    def test_mixed_types(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 1), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('d', 'c'), (1.0000000001, 'a')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 1.0

    def test_empty_lists(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 1.0

    def test_empty_vs_non_empty(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[('a',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 0.0

    def test_nan_and_string_mismatch(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[(math.nan,)]]
        orchestrator_input["predicted_queries"] = [[('NaN',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 0.0

    def test_float_outside_epsilon(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[(1.0,)]]
        orchestrator_input["predicted_queries"] = [[(1.00001,)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 0.0  # assuming default epsilon=1e-6

    def test_different_tuple_lengths(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b')]]
        orchestrator_input["predicted_queries"] = [[('a',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 0.0

    def test_no_matching_tuples(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('x', 'y'), ('z', 'w')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 0.0

        orchestrator_input.predicted_queries = [[('a', 'y'), ('c', 'w')]]
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 0.0

    def test_all_matching_tuples(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 1.0

    def test_partial_matching_tuples(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('a', 'b'), ('a', 'b'), ('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 0.5

    def test_empty_tables(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 1.0

    def test_single_tuple_table(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b')]]
        orchestrator_input["predicted_queries"] = [[('a', 'b')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 1.0

    def test_no_tuples_in_prediction(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 0.0

    def test_no_tuples_in_target(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 0.0

    def test_duplicate_tuples_in_target(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 0.5

    def test_all_matching_tuples_diff_order(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('c', 'd'), ('a', 'b')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 1.0

        orchestrator_input.predicted_queries = [[('d', 'c'), ('b', 'a')]]
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['tuple_constraint'] == 1.0

    def test_special_case(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('1', '2'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('1', '2'), ('c', 'd')]]
        result = evaluator_orchestrator.invoke(OrchestratorInput(**orchestrator_input))
        assert result[0]['tuple_constraint'] == 1.0

        orchestrator_input["predicted_queries"] = [[('1', '2'), ('c', 'd'), ('a', 'b')]]
        result = evaluator_orchestrator.invoke(OrchestratorInput(**orchestrator_input))
        assert result[0]['tuple_constraint'] == 1.0

        orchestrator_input["target_queries"] = [[('1', '2'), ('c', None, math.nan)]]
        orchestrator_input["predicted_queries"] = [[('1', '2'), (None, math.nan, 'c'), ('a', 'b')]]
        result = evaluator_orchestrator.invoke(OrchestratorInput(**orchestrator_input))
        assert result[0]['tuple_constraint'] == 1.0

        orchestrator_input["target_queries"] = [[('1', '2'), ('1', 'd')]]
        orchestrator_input["predicted_queries"] = [[('2', '1'), ('d', '1'), ('a', 'b')]]
        result = evaluator_orchestrator.invoke(OrchestratorInput(**orchestrator_input))
        assert result[0]['tuple_constraint'] == 1.0