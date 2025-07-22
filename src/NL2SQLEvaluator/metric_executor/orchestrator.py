from enum import Enum
from typing import Optional

from langgraph.func import entrypoint, task
from pydantic import BaseModel, ConfigDict

from NL2SQLEvaluator.db_executor import BaseSQLDBExecutor
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.metric_executor.execution_accuracy import worker_execution_accuracy
from NL2SQLEvaluator.metric_executor.qatch_metrics import (
    worker_cell_precision,
    worker_cell_recall,
    worker_tuple_cardinality,
    worker_tuple_constraint,
    worker_tuple_order,
    worker_f1_score
)
from NL2SQLEvaluator.metric_executor.utils_value import Value


class Metrics(Enum):
    """Enum for different metrics used in evaluation."""
    EXECUTION_ACCURACY = "execution_accuracy"
    F1_SCORE = "f1_score"
    CELL_PRECISION = "cell_precision"
    CELL_RECALL = "cell_recall"
    TUPLE_CARDINALITY = "tuple_cardinality"
    TUPLE_CONSTRAINT = "tuple_constraint"
    TUPLE_ORDER = "tuple_order"


metric_functions = {
    Metrics.EXECUTION_ACCURACY: worker_execution_accuracy,
    Metrics.F1_SCORE: worker_f1_score,
    Metrics.CELL_PRECISION: worker_cell_precision,
    Metrics.CELL_RECALL: worker_cell_recall,
    Metrics.TUPLE_CARDINALITY: worker_tuple_cardinality,
    Metrics.TUPLE_CONSTRAINT: worker_tuple_constraint,
    Metrics.TUPLE_ORDER: worker_tuple_order
}


@task
def execute_queries(queries: list[str | list],
                    executor: BaseSQLDBExecutor, epsilon=1e-6) -> list[list[tuple[Value, ...]]]:
    """Execute a list of SQL queries using the provided executor."""
    if isinstance(queries[0], str):
        # Assume queries is already executed
        queries = executor.execute_multiple_query(queries)
    formatted_queries = [
        [
            tuple(Value(raw=v, epsilon=epsilon) for v in row)  # update all the values
            for row in query  # for each row in the query
        ]
        for query in queries  # for each query
    ]
    return formatted_queries


@task
def execute_metrics(executed_target: list, executed_predicted: list, metrics: list[Metrics]) -> dict[str, float]:
    results = {}
    for metric in metrics:
        results[metric.name.lower()] = metric_functions[metric](executed_target, executed_predicted)
    return {name: result.result() for name, result in results.items()}


class OrchestratorInput(BaseModel):
    target_queries: list[str | list[tuple]]
    predicted_queries: list[str | list[tuple]]
    executor: Optional[BaseSQLDBExecutor] = None
    metrics_to_calculate: list[str]
    model_config = ConfigDict(arbitrary_types_allowed=True)


@entrypoint()
def orchestrator_worker(
        input: OrchestratorInput
) -> list[dict[str, float]]:
    logger = get_logger(__name__, level="INFO")
    logger.info(
        f"Starting evaluation of {len(input.target_queries)} target vs {len(input.predicted_queries)} predicted queries with metrics: {input.metrics_to_calculate}"
    )
    logger.warning(f"Initializing class with epsilon 10e-6, float/int number will be considered equal if they differ less than epsilon.")
    metrics = [Metrics[metric.upper()] for metric in input.metrics_to_calculate]
    executed_targets = execute_queries(input.target_queries, input.executor)
    executed_predicteds = execute_queries(input.predicted_queries, input.executor)
    results = []
    for tar, pred in zip(executed_targets.result(), executed_predicteds.result()):
        results.append(execute_metrics(tar, pred, metrics))
    return [r.result() for r in results]
