from collections import defaultdict
from typing import Optional, Any, Sequence

from langgraph.func import task, entrypoint
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
    worker_f1_score,
)
from NL2SQLEvaluator.metric_executor.utils_value import Value
from NL2SQLEvaluator.orchestrator_state import SingleTask, AvailableMetrics

logger = get_logger(__name__, level="INFO")

# Map enum -> callable (sync or task-like)
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
    metrics_to_calculate: list[str | AvailableMetrics]
    save_executed_query_in_cache: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def _validate_metrics(cls, v: Sequence[str | AvailableMetrics]) -> list[AvailableMetrics]:
        out: list[AvailableMetrics] = []
        for m in v:
            if isinstance(m, AvailableMetrics):
                out.append(m)
            elif isinstance(m, str):
                key = m.strip().upper()
                try:
                    out.append(AvailableMetrics[key])
                except KeyError as e:
                    raise ValueError(
                        f"Unknown metric '{m}'. Valid: {[m.name.lower() for m in AvailableMetrics]}.") from e
            else:
                raise ValueError(f"Metric must be str or AvailableMetrics, got {type(m)}")
        return out

@entrypoint()
def evaluator_orchestrator(params: OrchestratorInput) -> list[dict[str, float]]:
    logger = get_logger(__name__, level="INFO")

    logger.info(
        f"Starting evaluation of {len(params.target_queries)} target vs "
        f"{len(params.predicted_queries)} predicted queries with metrics: {params.metrics_to_calculate}"
    )
    logger.warning("Initializing evaluation with epsilon = 1e-6; numbers within epsilon are treated as equal.")

    metrics: list[AvailableMetrics] = params.metrics_to_calculate

    # Single engine or None
    if not isinstance(params.executor, list):
        eng = params.executor
        tar_ = params.target_queries
        pred_ = params.predicted_queries

        results, executed_targets, executed_predicteds = _get_metrics_from_batch(tar_, pred_, eng, metrics)
        if eng is not None:
            _cache_pairs(eng, pred_, executed_predicteds, save=params.save_executed_query_in_cache)
            _cache_pairs(eng, tar_, executed_targets, save=params.save_executed_query_in_cache)
        return results

    # Multiple engines: group by engine url and process subsets
    eng2idxs: dict[str, list[int]] = defaultdict(list)
    engname2engine: dict[str, BaseSQLDBExecutor] = {}

    for i, eng in enumerate(params.executor):
        eng2idxs[eng.engine_url].append(i)
        engname2engine[eng.engine_url] = eng

    out: list[Optional[dict[str, float]]] = [None] * len(params.target_queries)

    for eng_name, idxs in eng2idxs.items():
        logger.info(f"Processing {len(idxs)} queries for engine {eng_name}")
        eng = engname2engine[eng_name]
        tar_subset = [params.target_queries[i] for i in idxs]
        pred_subset = [params.predicted_queries[i] for i in idxs]

        res, executed_targets, executed_predicteds = _get_metrics_from_batch(tar_subset, pred_subset, eng, metrics)

        for i_local, i_global in enumerate(idxs):
            out[i_global] = res[i_local]

        _cache_pairs(eng, pred_subset, executed_predicteds, save=params.save_executed_query_in_cache)
        _cache_pairs(eng, tar_subset, executed_targets, save=params.save_executed_query_in_cache)

    # type: ignore[return-value] — we assert all filled
    return [r if r is not None else {m.name.lower(): 0.0 for m in metrics} for r in out]


def _get_metrics_from_batch(
        tar_subset: list[str | list[tuple[Any, ...]]],
        pred_subset: list[str | list[tuple[Any, ...]]],
        eng: BaseSQLDBExecutor,
        metrics: list[AvailableMetrics],
        *,
        epsilon: float = 1e-6,
) -> tuple[list[dict[str, float]], list[list[tuple[Any, ...]]], list[list[tuple[Any, ...]]]]:
    executed_targets = _execute_or_passthrough(tar_subset, eng)
    executed_predicteds = _execute_or_passthrough(pred_subset, eng)

    if len(executed_targets) != len(executed_predicteds):
        raise ValueError(
            f"Targets and predicteds length mismatch: {len(executed_targets)} vs {len(executed_predicteds)}"
        )

    results: list[dict[str, float]] = [
        execute_metrics(t, p, metrics, epsilon=epsilon)
        for t, p in zip(executed_targets, executed_predicteds)
    ]

    return results, executed_targets, executed_predicteds


def _execute_or_passthrough(
        queries: list[str | list[tuple[Any, ...]]],
        executor: Optional[BaseSQLDBExecutor],
) -> list[list[tuple[Any, ...]]]:
    """
    If `queries` is a list of SQL strings -> execute them.
    If it's already a list of rows -> return as-is.
    """
    if not queries:
        return []

    first = queries[0]
    if isinstance(first, str):
        if executor is None:
            raise ValueError("executor=None but queries require execution (strings provided).")
        return executor.execute_multiple_query(queries)  # expected list[list[tuple]]
    else:
        # assume already executed rows
        return queries  # type: ignore[return-value]


def _as_value_rows(rows: list[tuple[Any, ...]], epsilon: float) -> list[tuple[Value, ...]]:
    return [tuple(Value(raw=v, epsilon=epsilon) for v in r) for r in rows]


def execute_metrics(
        executed_target: list[tuple[Any, ...]] | None,
        executed_predicted: list[tuple[Any, ...]] | None,
        metrics: list[AvailableMetrics | str],
        *,
        epsilon: float = 1e-6,
) -> dict[str, float]:
    """Compute all metrics for a single (target, predicted) pair of executed results."""
    metrics = {AvailableMetrics(m) if isinstance(m, str) else m for m in metrics}
    if executed_predicted is None or executed_target is None:
        if executed_target is None:
            logger.error('Target execution failed.')
        # If either execution failed, all metrics are zero for this pair
        return {m.name.lower(): 0.0 for m in metrics}

    t_rows = _as_value_rows(executed_target, epsilon)
    p_rows = _as_value_rows(executed_predicted, epsilon)

    future_results = {m: metric_functions[m](t_rows, p_rows) for m in metrics}

    results = {
        # Support both direct floats and task-like results
        m.name.lower(): future.result() if hasattr(future, "result") else float(future)
        for m, future in future_results.items()
    }

    return results


def _cache_pairs(
        eng: BaseSQLDBExecutor,
        queries_subset: list[str | list[tuple[Any, ...]]],
        executed_subset: list[list[tuple[Any, ...]]],
        *,
        save: bool,
) -> None:
    if not save or getattr(eng, "cache_db", None) is None:
        return

    # Only cache queries that were SQL strings to begin with (skip already-executed lists)
    to_cache_queries: list[str] = []
    to_cache_results: list[list[tuple[Any, ...]]] = []

    for q, r in zip(queries_subset, executed_subset):
        if isinstance(q, str):
            to_cache_queries.append(q)
            to_cache_results.append(r)

    if to_cache_queries:
        logger.debug("Storing executed queries in cache")
        eng.cache_db.insert_bulk_in_cache(
            db_ids=[eng.db_id] * len(to_cache_queries),
            queries=to_cache_queries,
            results=to_cache_results,
        )


@task()
def evaluator_worker(single_task: SingleTask) -> SingleTask:
    logger = get_logger(__name__, level="INFO")
    logger.debug(f"Starting evaluation for metrics={single_task.metrics}")
    logger.debug("Using epsilon = 1e-6 for numeric comparisons.")

    executed_metrics = execute_metrics(
        single_task.target_sql.executed,
        single_task.predicted_sql.executed,
        single_task.metrics,
    )
    return SingleTask(results=executed_metrics, **single_task.model_dump(exclude={"results"}))
