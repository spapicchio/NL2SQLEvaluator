from typing import Any

from NL2SQLEvaluator.evaluator_nodes.evaluator_input import (
    EvaluationType,
    EvalText2SQLTask,
    EvalAmbigText2SQLTask,
    EvalUnansText2SQLTask,
    EvalText2CypherTask,
    EvalText2SparqlTask,
    BaseEvalTask
)


class FactoryTaskEvaluator:
    """Factory for generating specialized Task objects.

        Why: Use this to decouple the pipeline's data loading from the specific
        validation logic of different metrics.

        How to use:
            >>> task = EvaluationTaskFactory.create(
            ...     task_type=EvaluationType.TEXT2SQL,
            ...     predictions=[...],
            ...     target=[...]
            ... )
        """

    _TASK_INPUT = {
        EvaluationType.TEXT2SQL: EvalText2SQLTask,
        EvaluationType.AMBIGTEXT2SQL: EvalAmbigText2SQLTask,
        EvaluationType.UNANSTEXT2SQL: EvalUnansText2SQLTask,
        EvaluationType.TEXT2CYPHER: EvalText2CypherTask,
        EvaluationType.TEXT2SPARQL: EvalText2SparqlTask,
    }

    @classmethod
    def create(cls, task_type: EvaluationType, predictions, target, **kwargs: Any) -> BaseEvalTask:
        """Creates a task instance based on the provided type.

        Args:
            target: The ground truth data for evaluation.
            predictions: The model's output to be evaluated.
            task_type: The category of evaluation to perform.
            **kwargs: Arguments passed to the Pydantic model (predictions, target, etc).

        Returns:
            An instance of a specialized EvalTask.
        """
        task_type = EvaluationType(task_type)  # Ensure task_type is an instance of EvaluationType
        if task_type not in cls._TASK_INPUT:
            raise ValueError(
                f"Unsupported task type: {task_type}. Supported types are: {list(cls._TASK_INPUT.keys())}"
            )

        model_class = cls._TASK_INPUT[task_type]
        return model_class(predictions=predictions, target=target, **kwargs)
