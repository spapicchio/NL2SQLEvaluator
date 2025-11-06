from NL2SQLEvaluator.evaluator_nodes.evaluator_protocol import EvaluateTask
from NL2SQLEvaluator.evaluator_nodes.utils import get_majority_voting_values
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import register_node

logger = get_logger(__name__)


@register_node()
class BirdEXEvaluator:
    """Implementation of the standard EX from BIRD: https://bird-bench.github.io/"""

    def execute_metric(
            self,
            tasks: list[EvaluateTask],
            *args,
            **kwargs
    ) -> list[float]:
        # self._validate_inputs(multiple_tasks_preds, multiple_tasks_tar)
        results = [self._ex(task) for task in tasks]
        return results

    def _ex(self, task: EvaluateTask) -> float:
        if len(task.target) == 0 and len(task.predictions) == 0:
            logger.warning('no target and no predictions provided for EX evaluation, returning 1.0')
            return 1.0
        if len(task.target) == 0:
            logger.warning('No target provided for EX evaluation, returning 0.0')
            return 0.0

        target = frozenset(task.target[0])
        majority_vote = get_majority_voting_values(task.predictions, count_cardinality_in_row=False)
        if majority_vote is None:
            return 0.0
        return 1.0 if frozenset(target) == majority_vote else 0.0

    @staticmethod
    def _validate_inputs(multiple_tasks_preds, multiple_tasks_tar) -> None:
        if not (len(multiple_tasks_preds) == len(multiple_tasks_tar)):
            raise ValueError(
                "Lengths must match: target_queries, llm_predictions, db_files."
            )
        for ref in multiple_tasks_tar:
            if not len(ref) == 1:
                raise ValueError(
                    "For Execution Accuracy the target must have len 1. Otherwise use Ambiguous Execution Accuracy."
                )
