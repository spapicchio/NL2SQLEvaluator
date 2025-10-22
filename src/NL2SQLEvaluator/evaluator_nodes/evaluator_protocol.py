from typing import Protocol, TypeAlias

from NL2SQLEvaluator.db_executors_nodes.db_executor_protocol import OutputTable

TaskPredictions: TypeAlias = list[OutputTable]
TaskTargets: TypeAlias = list[OutputTable]


class EvaluatorProtocol(Protocol):
    def execute_metric(
            self,
            multiple_tasks_preds: list[TaskPredictions],
            multiple_tasks_tar: list[TaskTargets],
            *args,
            **kwargs
    ) -> list[float]:
        """
        Each element in predictions and references corresponds to one input example.
        Each prediction or reference is a list of (multiple) output tables.
        """
        ...
