from collections import Counter

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import OutputTable


def sort_with_different_types(arr: tuple) -> tuple:
    return tuple(sorted(arr, key=sort_key))


def sort_key(x):
    raw = x
    if raw is None:
        return 0, ''
    elif isinstance(raw, (int, float)):
        return 1, float(raw)
    else:
        return 2, str(raw)


def get_majority_voting_values(values: list[OutputTable], count_cardinality_in_row: bool) -> OutputTable | None:
    counter = Counter()
    for pred_n in values:
        if pred_n is None or isinstance(pred_n, Exception):
            continue

        if count_cardinality_in_row:
            key = tuple([sort_with_different_types(row) for row in pred_n])

        else:
            key = frozenset(pred_n)

        counter[key] += 1

    if not counter:
        # case where all the predictions are None
        return None

    majority_vote, _ = counter.most_common(1)[0]
    return majority_vote
