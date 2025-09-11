from collections import defaultdict
from typing import Optional, Any

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
from NL2SQLEvaluator.orchestrator_state import SingleTask, AvailableMetrics

metric_functions = {
    AvailableMetrics.EXECUTION_ACCURACY: worker_execution_accuracy,
    AvailableMetrics.F1_SCORE: worker_f1_score,
    AvailableMetrics.CELL_PRECISION: worker_cell_precision,
    AvailableMetrics.CELL_RECALL: worker_cell_recall,
    AvailableMetrics.TUPLE_CARDINALITY: worker_tuple_cardinality,
    AvailableMetrics.TUPLE_CONSTRAINT: worker_tuple_constraint,
    AvailableMetrics.TUPLE_ORDER: worker_tuple_order
}


class OrchestratorInput(BaseModel):
    target_queries: list[str | list[tuple]]
    predicted_queries: list[str | list[tuple]]
    executor: Optional[BaseSQLDBExecutor | list[BaseSQLDBExecutor]] = None
    metrics_to_calculate: list[str]
    model_config = ConfigDict(arbitrary_types_allowed=True)


@entrypoint()
def evaluator_orchestrator(
        input: OrchestratorInput
) -> list[dict[str, float]]:
    def run_db_id_queries(tar_, pred_, engine):
        executed_targets = execute_multiple_queries(tar_, engine)
        executed_predicteds = execute_multiple_queries(pred_, engine)
        results = []
        for tar, pred in zip(executed_targets.result(), executed_predicteds.result()):
            results.append(execute_metrics(tar, pred, metrics))
        return [r.result() for r in results]

    logger = get_logger(__name__, level="INFO")
    logger.info(
        f"Starting evaluation of {len(input.target_queries)} target vs {len(input.predicted_queries)} predicted queries with metrics: {input.metrics_to_calculate}"
    )
    logger.warning(
        f"Initializing class with epsilon 10e-6, float/int number will be considered equal if they differ less than epsilon.")
    metrics = [AvailableMetrics[metric.upper()] for metric in input.metrics_to_calculate]
    if isinstance(input.executor, list):
        eng2idxs = defaultdict(list)
        eng_name2engine = {}
        for i, eng in enumerate(input.executor):
            eng2idxs[eng.engine_url].append(i)
            eng_name2engine[eng.engine_url] = eng

        results = [None] * len(input.target_queries)
        for eng_name, idxs in eng2idxs.items():
            logger.info(f"Processing {len(idxs)} queries for engine {eng_name}")
            tar_ = [input.target_queries[i] for i in idxs]
            pred_ = [input.predicted_queries[i] for i in idxs]
            eng = eng_name2engine[eng_name]
            res = run_db_id_queries(tar_, pred_, eng)
            for i, r in zip(idxs, res):
                results[i] = r
        return results
    else:
        return run_db_id_queries(input.target_queries, input.predicted_queries, input.executor)


@task()
def execute_multiple_queries(queries: list[str | list],
                             executor: BaseSQLDBExecutor, epsilon=1e-6) -> list[list[tuple[Any]]]:
    """Execute a list of SQL queries using the provided executor."""
    if isinstance(queries[0], str):
        # Assume queries is already executed
        queries = executor.execute_multiple_query(queries)
    return queries


@task()
def execute_metrics(executed_target: list, executed_predicted: list, metrics: list[AvailableMetrics], epsilon=1e-6) -> \
        dict[
            str, float]:
    results = {}

    if executed_predicted is None:
        # Execution failed, return 0 for all metrics
        return {name.value: 0.0 for name in metrics}

    for metric in metrics:
        results[metric.name.lower()] = metric_functions[metric](
            [tuple(Value(raw=v, epsilon=epsilon) for v in row) for row in executed_target],
            [tuple(Value(raw=v, epsilon=epsilon) for v in row) for row in executed_predicted],
        )
    return {name: result.result() for name, result in results.items()}


@task()
def evaluator_worker(
        single_task: SingleTask
) -> SingleTask:
    logger = get_logger(__name__, level="INFO")
    logger.debug(
        f"Starting evaluation for {single_task.metrics}"
    )
    logger.debug(
        f"Initializing class with epsilon 10e-6, float/int number will be considered equal if they differ less than epsilon."
    )
    executed_metrics = execute_metrics(
        single_task.target_sql.executed,
        single_task.predicted_sql.executed,
        single_task.metrics).result()
    return SingleTask(results=executed_metrics, **single_task.model_dump(exclude={"results"}))

    # TODO: Check different data types coming from different database! Probably best option is to pass everything as str
    # CHeck if when executing obtaining different types of data str instead to float
    # Check if MySQL stores same results of executing the query
