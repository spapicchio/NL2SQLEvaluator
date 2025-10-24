from NL2SQLEvaluator.evaluator_nodes.evaluator_protocol import SingleTaskPred, SingleTaskTar
from NL2SQLEvaluator.evaluator_nodes.utils import get_majority_voting_values
from NL2SQLEvaluator.node_registry import register_node


@register_node()
class BirdEXEvaluator:
    """Implementation of the standard EX from BIRD: https://bird-bench.github.io/"""

    def execute_metric(
            self,
            multiple_tasks_preds: list[SingleTaskPred],
            multiple_tasks_tar: list[SingleTaskTar],
            *args,
            **kwargs
    ) -> list[float]:
        self._validate_inputs(multiple_tasks_preds, multiple_tasks_tar)
        results = [self._ex(pred, tar) for pred, tar in zip(multiple_tasks_preds, multiple_tasks_tar)]
        return results

    def _ex(self, task_pred: SingleTaskPred, task_tar: SingleTaskTar) -> float:
        target = frozenset(task_tar[0])
        majority_vote = get_majority_voting_values(task_pred, count_cardinality_in_row=False)
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
