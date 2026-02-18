from typing import Protocol

from NL2SQLEvaluator.db_executor_nodes.db_executor_output import GenericExecutorOutput
from NL2SQLEvaluator.evaluator_nodes.evaluator_input import BaseEvalTask, EvaluationType
from NL2SQLEvaluator.evaluator_nodes.factory_evaluator import FactoryTaskEvaluator

TargetType = int | GenericExecutorOutput | list[GenericExecutorOutput]
PredType = str | list[GenericExecutorOutput]


class EvaluatorProtocol(Protocol):
    def execute_metric(
            self,
            tasks: list[BaseEvalTask],
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
        targets: list[TargetType],
        predictions: list[PredType],
        task_type: EvaluationType,
        *args,
        **kwargs
) -> list[float]:
    """
    Evaluate a single pair of predictions and targets using the provided evaluator.
    """
    tasks = [
        FactoryTaskEvaluator.create(
            task_type=task_type,
            target=target,
            predictions=prediction
        )
        for target, prediction in zip(targets, predictions)
    ]
    scores = evaluator.execute_metric(tasks, *args, **kwargs)
    return scores
