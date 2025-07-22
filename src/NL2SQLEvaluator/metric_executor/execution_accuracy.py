from langgraph.func import task

from NL2SQLEvaluator.metric_executor.utils_value import Value


@task
def worker_execution_accuracy(
        executed_target: list[tuple[Value, ...]], executed_predicted: list[tuple[Value, ...]]
) -> float:
    """Calculate the execution accuracy between target and predicted executed queries."""
    if len(executed_target) == len(executed_predicted) == 0:
        return 1.0
    if len(executed_target) != len(executed_predicted):
        return 0.0

    gold_row_set = {
        tuple(sort_with_different_types(gold_row)) for gold_row in executed_target
    }
    pred_row_set = {
        tuple(sort_with_different_types(predicted_row)) for predicted_row in executed_predicted
    }

    return float(gold_row_set == pred_row_set)


def sort_with_different_types(arr: tuple[Value, ...]) -> tuple[Value, ...]:
    return tuple(sorted(arr, key=sort_key))


def sort_key(x: Value):
    raw = x.raw
    if raw is None:
        return 0, ''
    elif isinstance(raw, (int, float)):
        return 1, float(raw)
    else:
        return 2, str(raw)
