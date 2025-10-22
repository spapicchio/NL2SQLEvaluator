from langgraph.func import task

from NL2SQLEvaluator.metric_executor.utils_value import Value


@task()
def worker_ex(
        executed_target: list[tuple[Value, ...]], executed_predicted: list[tuple[Value, ...]]
) -> float:
    """Calculate the execution accuracy between target and predicted executed queries."""
    if len(executed_target) == len(executed_predicted) == 0:
        return 1.0
    return float(set(executed_target) == set(executed_predicted))
