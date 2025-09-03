import math

import pytest

from NL2SQLEvaluator.metric_executor.evaluator_orchestrator import evaluator_orchestrator, OrchestratorInput


class TestCellRecall:
    @pytest.fixture
    def orchestrator_input(self):
        return {
            "executor": None,
            "metrics_to_calculate": ["cell_recall"]
        }

    def test_equal(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 1.0

    def test_different(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 0.5

    def test_equal_but_different_projection(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('b', 'a'), ('d', 'c')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 1.0

    def test_equal_but_different_tuple_order(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('d', 'c'), ('b', 'a')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 1.0

    def test_null_values(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', None), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('d', 'c'), (None, 'a')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 1.0

    def test_null_math_none(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', math.nan), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('d', 'c'), (math.nan, 'a')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 1.0

    def test_mixed_types(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 1), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('d', 'c'), (1.0000000001, 'a')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 1.0

    def test_empty_lists(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 1.0

    def test_empty_vs_non_empty(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[('a',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 0.0

    def test_nan_and_string_mismatch(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[(math.nan,)]]
        orchestrator_input["predicted_queries"] = [[('NaN',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 0.0

    def test_float_outside_epsilon(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[(1.0,)]]
        orchestrator_input["predicted_queries"] = [[(1.00001,)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 0.0  # assuming default epsilon=1e-6

    def test_different_tuple_lengths(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b')]]
        orchestrator_input["predicted_queries"] = [[('a',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 0.5

    def test_no_matching_cells(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('x', 'y'), ('z', 'w')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 0.0

    def test_all_matching_cells(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 1.0

    def test_partial_matching_cells(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('a', 'x'), ('y', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 0.5

    def test_empty_tables(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 1.0

    def test_single_cell_table(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a',)]]
        orchestrator_input["predicted_queries"] = [[('a',)]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 1.0

    def test_no_cells_in_prediction(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 0.0

    def test_no_cells_in_target(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[]]
        orchestrator_input["predicted_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 0.0

    def test_duplicate_cells_in_prediction(self, orchestrator_input):
        orchestrator_input["target_queries"] = [[('a', 'b'), ('c', 'd')]]
        orchestrator_input["predicted_queries"] = [[('a', 'a'), ('b', 'b'), ('c', 'd')]]
        orchestrator_input = OrchestratorInput(**orchestrator_input)
        result = evaluator_orchestrator.invoke(orchestrator_input)
        assert result[0]['cell_recall'] == 1.0

    def test_special_chatgpt_case(self, orchestrator_input):
        target_data = [[
            ('573585', '10/31/2019', '22282', '12-Egg-House-Painted-Wood', 35.83, 2, 14585.0, 'United-Kingdom')
        ]]
        orchestrator_input["target_queries"] = target_data
        orchestrator_input["predicted_queries"] = [[('573585',)]]

        result = evaluator_orchestrator.invoke(OrchestratorInput(**orchestrator_input))
        assert result[0]['cell_recall'] == round(1 / len(target_data[0][0]), 3)
        target_data = [[
            ('False', 'BC', 18, 0.8, 'AB', 'CA', 99, 5.4796861662000005, 'windows', 'Paid'),
            ('True', 'BC', 18, 0.8, 'AB', 'CA', 157, 9.7706304446, 'linux', 'Paid')
        ]]

        orchestrator_input["target_queries"] = target_data
        orchestrator_input["predicted_queries"] = [[(18,), (18,)]]
        result = evaluator_orchestrator.invoke(OrchestratorInput(**orchestrator_input))
        # 14 distinct elements in target
        assert result[0]['cell_recall'] == round(1 / 14, 3)
