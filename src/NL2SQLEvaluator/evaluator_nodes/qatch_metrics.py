from collections import Counter
from enum import Enum

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import OutputTable
from NL2SQLEvaluator.evaluator_nodes.evaluator_protocol import EvaluateTask
from NL2SQLEvaluator.evaluator_nodes.utils import sort_with_different_types, get_majority_voting_values
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import register_node

logger = get_logger(__name__)


class QatchMetric(Enum):
    """Enum for different metrics used in evaluation."""
    CELL_PRECISION = "cell_precision"
    CELL_RECALL = "cell_recall"
    TUPLE_CARDINALITY = "tuple_cardinality"
    TUPLE_CONSTRAINT = "tuple_constraint"
    TUPLE_ORDER = "tuple_order"
    CP_CR_TC = "cp_cr_tc"
    F1_SCORE = "f1_score"

    # Optional: accept case-insensitive and aliases like "val" / "validation"
    _ALIASES = {
        'f1_score': 'F1_SCORE',
        'cell_precision': 'CELL_PRECISION',
        'cell_recall': 'CELL_RECALL',
        'tuple_cardinality': 'TUPLE_CARDINALITY',
        'tuple_constraint': 'TUPLE_CONSTRAINT',
        'tuple_order': 'TUPLE_ORDER',
        'cp_cr_tc': 'CP_CR_TC',
    }

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            v = value.strip().lower()
            v = cls._ALIASES.get(v, v)
            for m in cls:
                if m.value == v or m.name == v:
                    return m
        return None


@register_node()
class QATCHEvaluator:
    def execute_metric(
            self,
            tasks: list[EvaluateTask],
            metric: QatchMetric | str,
            *args,
            **kwargs
    ) -> list[float]:
        results = []
        metric = QatchMetric(metric)
        for task in tasks:
            if len(task.target) == 0:
                logger.warning('No target provided for EX evaluation, returning 0.0')
                results.append(0.0)
                continue

            tar = task.target[0]
            pred = task.predictions

            tar = OutputTable(rows=[tuple(sort_with_different_types(row)) for row in tar])
            majority_vote = get_majority_voting_values(pred, count_cardinality_in_row=True)

            if majority_vote is None:
                results.append(0.0)
            elif len(majority_vote) == len(tar) == 0:
                results.append(1.0)
            elif len(majority_vote) != 0 and len(tar) == 0 or len(majority_vote) == 0 and len(tar) != 0:
                results.append(0.0)

            elif metric == QatchMetric.TUPLE_CARDINALITY:
                results.append(self._tuple_cardinality(majority_vote, tar))
            elif metric == QatchMetric.TUPLE_CONSTRAINT:
                results.append(self._tuple_constraint(majority_vote, tar))
            elif metric == QatchMetric.CELL_PRECISION:
                results.append(self._cell_precision(majority_vote, tar))
            elif metric == QatchMetric.CELL_RECALL:
                results.append(self._cell_recall(majority_vote, tar))
            elif metric == QatchMetric.TUPLE_ORDER:
                results.append(self._tuple_order(majority_vote, tar))
            elif metric == QatchMetric.CP_CR_TC:
                results.append(self._cp_cr_tc(majority_vote, tar))
            elif metric == QatchMetric.F1_SCORE:
                results.append(self._f1_score(majority_vote, tar))
            else:
                raise NotImplementedError(f"The metric {metric} is not implemented in QATCHEvaluator.")

        return results

    def _tuple_cardinality(self, task_pred: OutputTable, task_tar: OutputTable) -> float:

        if len(task_pred) >= len(task_tar):
            # in case we have more elements in the prediction than in the target
            return round(len(task_tar) / len(task_pred), 3)

            # in case we have more elements in the target than in the prediction
        return round(len(task_pred) / len(task_tar), 3)

    def _tuple_constraint(self, task_pred: OutputTable, task_tar: OutputTable):
        target_len = len(task_tar)
        prediction_len = len(task_pred)
        if target_len == prediction_len == 0:
            return 1.0
        if prediction_len != 0 and target_len == 0 or prediction_len == 0 and target_len != 0:
            return 0.0

        # When comparing tuples, the projection orders do not matter (Name, Surname) = (Surname, Name)

        count_targ_dict = Counter(task_tar)
        count_pred_dict = Counter(task_pred)

        cardinality = [count_pred_dict[key] == count for key, count in count_targ_dict.items()]

        return round(sum(cardinality) / len(cardinality), 3)

    def _cell_precision(self, task_pred: OutputTable, task_tar: OutputTable) -> float:
        from itertools import chain

        prediction = set(chain.from_iterable(task_pred))
        target = set(chain.from_iterable(task_tar))
        intersected_cells = target.intersection(prediction)

        return round(len(intersected_cells) / len(prediction), 3) if len(prediction) > 0 else 0.0

    def _cell_recall(self, task_pred: OutputTable, task_tar: OutputTable) -> float:
        from itertools import chain

        prediction = set(chain.from_iterable(task_pred))
        target = set(chain.from_iterable(task_tar))
        intersected_cells = target.intersection(prediction)

        return round(len(intersected_cells) / len(target), 3) if len(target) > 0 else 0.0

    def _tuple_order(self, task_pred: OutputTable, task_tar: OutputTable) -> float:
        import numpy as np
        def normalize(data: float):
            data = [-1, data, 1]
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            return data[1]

        new_pred = []
        [new_pred.append(row) for row in task_pred
         if row in task_tar and row not in new_pred]

        # same for target
        new_target = []
        [new_target.append(row) for row in task_tar
         if row in task_pred and row not in new_target]

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

    def _f1_score(self, task_pred: OutputTable, task_tar: OutputTable) -> float:
        cell_precision = self._cell_precision(task_pred, task_tar)
        cell_recall = self._cell_recall(task_pred, task_tar)
        if cell_precision + cell_recall == 0:
            return 0.0

        f1_score = 2 * (cell_precision * cell_recall) / (cell_precision + cell_recall)
        return float(f1_score)

    def _cp_cr_tc(self, task_pred: OutputTable, task_tar: OutputTable) -> float:
        cell_precision = self._cell_precision(task_pred, task_tar)
        cell_recall = self._cell_recall(task_pred, task_tar)
        tuple_cardinality = self._tuple_cardinality(task_pred, task_tar)

        combined_score = (cell_precision + cell_recall + tuple_cardinality) / 3
        return round(combined_score, 3)

    @staticmethod
    def _validate_inputs(multiple_tasks_preds, multiple_tasks_tar) -> None:
        if not (len(multiple_tasks_preds) == len(multiple_tasks_tar)):
            raise ValueError(
                "Lengths must match: target_queries, llm_predictions, db_files."
            )
        for ref in multiple_tasks_tar:
            if not len(ref) == 1:
                raise ValueError(
                    "For Execution Accuracy the target must have len 1. Otherwise use Ambiguous Execution Accuracy."
                )
