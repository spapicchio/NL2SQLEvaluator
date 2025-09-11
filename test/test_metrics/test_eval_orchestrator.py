import pytest

from NL2SQLEvaluator.metric_executor.evaluator_orchestrator import OrchestratorInput
from NL2SQLEvaluator.orchestrator_state import AvailableDialect, AvailableMetrics
from NL2SQLEvaluator.utils import utils_get_engine


class TestEvaluatorOrchestrator:
    @pytest.fixture
    def executor(self):
        return utils_get_engine(
            relative_base_path='data/bird_dev/dev_databases',
            db_executor=AvailableDialect.sqlite,
            db_id='california_schools',
        )

    @pytest.fixture
    def orchestrator_input(self, executor):
        return OrchestratorInput(
            target_queries=[
                'SELECT * FROM frpm;',
                """SELECT frpm.CDSCode
                   FROM frpm
                   WHERE CDSCode == '01100170109835';"""
            ],
            predicted_queries=['SELECT * FROM broken;', 'SELECT * FROM frpm;'],
            executor=[executor, executor],
            metrics_to_calculate=[AvailableMetrics.EXECUTION_ACCURACY, AvailableMetrics.F1_SCORE],
        )

    def test_orchestrator(self, orchestrator_input):
        from NL2SQLEvaluator.metric_executor.evaluator_orchestrator import evaluator_orchestrator

        results = evaluator_orchestrator.invoke(orchestrator_input)

        assert 'execution_accuracy' in results[0] and 'f1_score' in results[0]
        assert results[0]['execution_accuracy'] == 0.0
        assert results[0]['f1_score'] == 0.0

        assert 'execution_accuracy' in results[1] and 'f1_score' in results[1]
