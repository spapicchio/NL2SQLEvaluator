from collections import Counter
from itertools import chain

import numpy as np
from langgraph.func import task

from NL2SQLEvaluator.metric_executor.utils_value import Value, sort_with_different_types


def _base_for_precision_and_recall(executed_target: list[tuple[Value, ...]],
                                   executed_predicted: list[tuple[Value, ...]]
                                   ):
    """Base function to get target and prediction sets."""
    target = set(chain.from_iterable(executed_target))
    prediction = set(chain.from_iterable(executed_predicted))
    intersected_cells = target.intersection(prediction)
    sum_cell_match = len(intersected_cells)

    return sum_cell_match, len(target), len(prediction)


@task()
def worker_cell_precision(executed_target: list[tuple[Value, ...]],
                          executed_predicted: list[tuple[Value, ...]]) -> float:
    target_len = len(executed_target)
    prediction_len = len(executed_predicted)
    if target_len == prediction_len == 0:
        return 1.0

    if target_len == prediction_len == 0:
        return 1.0
    if prediction_len != 0 and target_len == 0 or prediction_len == 0 and target_len != 0:
        return 0.0

    sum_cell_match, len_target, len_prediction = _base_for_precision_and_recall(executed_target, executed_predicted)

    return round(sum_cell_match / len_prediction, 3)


@task()
def worker_cell_recall(executed_target: list[tuple[Value, ...]], executed_predicted: list[tuple[Value, ...]]) -> float:
    target_len = len(executed_target)
    prediction_len = len(executed_predicted)

    if target_len == prediction_len == 0:
        return 1.0

    if target_len == prediction_len == 0:
        return 1.0

    if prediction_len != 0 and target_len == 0 or prediction_len == 0 and target_len != 0:
        return 0.0

    sum_cell_match, len_target, len_prediction = _base_for_precision_and_recall(executed_target, executed_predicted)
    return round(sum_cell_match / len_target, 3)


@task()
def worker_tuple_cardinality(executed_target: list[tuple[Value, ...]],
                             executed_predicted: list[tuple[Value, ...]]) -> float:
    if len(executed_target) == len(executed_predicted) == 0:
        return 1.0

    if len(executed_predicted) >= len(executed_target):
        # in case we have more elements in the prediction than in the target
        return round(len(executed_target) / len(executed_predicted), 3)

        # in case we have more elements in the target than in the prediction
    return round(len(executed_predicted) / len(executed_target), 3)


@task()
def worker_tuple_constraint(executed_target: list[tuple[Value, ...]],
                            executed_predicted: list[tuple[Value, ...]]) -> float:
    target_len = len(executed_target)
    prediction_len = len(executed_predicted)
    if target_len == prediction_len == 0:
        return 1.0
    if prediction_len != 0 and target_len == 0 or prediction_len == 0 and target_len != 0:
        return 0.0

    # When comparing tuples, the projection orders do not matter (Name, Surname) = (Surname, Name)
    target = [tuple(sort_with_different_types(row)) for row in executed_target]
    prediction = [tuple(sort_with_different_types(row)) for row in executed_predicted]

    count_targ_dict = Counter(target)
    count_pred_dict = Counter(prediction)

    cardinality = [count_pred_dict[key] == count for key, count in count_targ_dict.items()]

    return round(sum(cardinality) / len(cardinality), 3)


@task()
def worker_tuple_order(executed_target: list[tuple[Value, ...]], executed_predicted: list[tuple[Value, ...]]) -> float:
    def normalize(data: float):
        data = [-1, data, 1]
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return data[1]

    target_len = len(executed_target)
    prediction_len = len(executed_predicted)
    if target_len == prediction_len == 0:
        return 1.0
    if prediction_len != 0 and target_len == 0 or prediction_len == 0 and target_len != 0:
        return 0.0

    executed_target =  [tuple(sort_with_different_types(gold_row)) for gold_row in executed_target]
    executed_predicted = [tuple(sort_with_different_types(predicted_row)) for predicted_row in executed_predicted]

    # take only prediction that are in target without duplicates
    # MAINTAINING the order
    new_pred = []
    [new_pred.append(pred) for pred in executed_predicted
     if pred in executed_target and pred not in new_pred]
    # same for target
    new_target = []
    [new_target.append(tar) for tar in executed_target
     if tar in executed_predicted and tar not in new_target]

    if len(new_target) == 0:
        # case when prediction does not have any element in target
        rho = 0.0

    else:
        target_ranks = [i for i in range(len(new_target))]
        pred_ranks = [new_target.index(row) for row in new_pred]

        diff_rank_squared = [(tar - pred) ** 2
                             for tar, pred in zip(target_ranks, pred_ranks)]

        sum_diff_rank_squared = sum(diff_rank_squared)

        n = len(new_target) if len(new_target) > 1 else 2
        rho = 1 - 6 * sum_diff_rank_squared / (n * (n ** 2 - 1))

    return float(normalize(round(rho, 3)))


@task()
def worker_f1_score(executed_target: list[tuple[Value, ...]], executed_predicted: list[tuple[Value, ...]]) -> float:
    cell_precision = worker_cell_precision(executed_target, executed_predicted)
    cell_recall = worker_cell_recall(executed_target, executed_predicted)
    cell_precision = cell_precision.result()
    cell_recall = cell_recall.result()
    if cell_precision + cell_recall == 0:
        return 0.0

    # Calculate F1 score
    f1_score = 2 * (cell_precision * cell_recall) / (cell_precision + cell_recall)
    return float(f1_score)
