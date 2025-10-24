from typing import Protocol, TypeAlias

from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import OutputTable, ExecutorError

SingleTaskPred: TypeAlias = list[OutputTable]
SingleTaskTar: TypeAlias = list[OutputTable]


class EvaluatorProtocol(Protocol):
    def execute_metric(
            self,
            multiple_tasks_preds: list[SingleTaskPred],
            multiple_tasks_tar: list[SingleTaskTar],
            *args,
            **kwargs
    ) -> list[float]:
        """
        Each element in predictions and references corresponds to one input example.
        Each prediction or reference is a list of (multiple) output tables.
        """
        ...


def evaluate_target_and_pred(
        evaluator: EvaluatorProtocol,
        multiple_tasks_preds: list[list[OutputTable | ExecutorError]],
        multiple_tasks_tar: list[list[OutputTable | ExecutorError]],
        *args,
        **kwargs
) -> list[float]:
    """
    Evaluate a single pair of predictions and targets using the provided evaluator.
    """

    filtered_preds = []
    filtered_tars = []
    for preds, tars in zip(multiple_tasks_preds, multiple_tasks_tar):
        filtered_task_preds = [pred for pred in preds if not isinstance(pred, ExecutorError)]
        filtered_preds.append(filtered_preds) if len(filtered_task_preds) != 0 else filtered_preds.append([])

        filtered_task_tars = [tar for tar in tars if not isinstance(tar, ExecutorError)]
        filtered_tars.append(filtered_tars) if len(filtered_task_tars) != 0 else filtered_tars.append([])

    scores = evaluator.execute_metric(
        multiple_tasks_preds=filtered_preds,
        multiple_tasks_tar=filtered_tars,
        *args,
        **kwargs
    )
    return scores
