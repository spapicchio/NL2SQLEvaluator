from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import OutputTable
from NL2SQLEvaluator.evaluator_nodes.evaluator_protocol import EvaluateTask
from NL2SQLEvaluator.evaluator_nodes.utils import get_majority_voting_values, sort_with_different_types
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import register_node

logger = get_logger(__name__)


@register_node()
class EXEvaluator:
    def execute_metric(
            self,
            tasks: list[EvaluateTask],
            *args,
            **kwargs
    ) -> list[float]:
        results = [self._ex(task) for task in tasks]
        return results

    def _ex(self, task: EvaluateTask) -> float:
        if len(task.target) == 0 and len(task.predictions) == 0:
            logger.warning('No target and no predictions provided for EX evaluation, returning 1.0')
            return 1.0
        if len(task.target) == 0:
            logger.warning('No target provided for EX evaluation, returning 0.0')
            return 0.0

        target = OutputTable(rows=[tuple(sort_with_different_types(row)) for row in task.target[0]])
        pred = get_majority_voting_values(task.predictions, count_cardinality_in_row=True)

        if pred is None:
            return 0.0
        if len(target) == len(pred) == 0:
            return 1.0
        if len(target) != len(pred):
            return 0.0

        return float(set(target) == set(pred))
