from NL2SQLEvaluator.db_readers_nodes.db_reader_protocol import OutputTable
from NL2SQLEvaluator.evaluator_nodes.evaluator_protocol import TaskPredictions, TaskTargets
from NL2SQLEvaluator.evaluator_nodes.utils import get_majority_voting_values, sort_with_different_types
from NL2SQLEvaluator.node_registry import register_node


@register_node()
class EXEvaluator:
    def execute_metric(
            self,
            multiple_tasks_preds: list[TaskPredictions],
            multiple_tasks_tar: list[TaskTargets],
            *args,
            **kwargs
    ) -> list[float]:
        self._validate_inputs(multiple_tasks_preds, multiple_tasks_tar)
        results = [self._ex(pred, tar) for pred, tar in zip(multiple_tasks_preds, multiple_tasks_tar)]
        return results

    def _ex(self, task_pred: TaskPredictions, task_tar: TaskTargets) -> float:

        target: OutputTable = [tuple(sort_with_different_types(row)) for row in task_tar[0]]
        pred = get_majority_voting_values(task_pred, count_cardinality_in_row=True)

        if pred is None:
            return 0.0
        if len(target) == len(pred) == 0:
            return 1.0
        if len(target) != len(pred):
            return 0.0

        return float(set(target) == set(pred))

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
