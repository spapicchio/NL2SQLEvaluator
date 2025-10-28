from typing import Protocol

from pydantic import BaseModel

from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecutorError
from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import OutputTable


class EvaluateTask(BaseModel):
    predictions: list[OutputTable]
    target: list[OutputTable]


class EvaluatorProtocol(Protocol):
    def execute_metric(
            self,
            tasks: list[EvaluateTask],
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
        multiple_tasks_tars: list[list[OutputTable | ExecutorError]],
        *args,
        **kwargs
) -> list[float]:
    """
    Evaluate a single pair of predictions and targets using the provided evaluator.
    """
    tasks = [
        EvaluateTask(
            predictions=[pred for pred in preds if isinstance(pred, OutputTable)],
            target=[tar for tar in tars if isinstance(tar, OutputTable)]
        )
        for preds, tars in zip(multiple_tasks_preds, multiple_tasks_tars)
    ]
    scores = evaluator.execute_metric(
        tasks=tasks,
        *args,
        **kwargs
    )
    return scores
