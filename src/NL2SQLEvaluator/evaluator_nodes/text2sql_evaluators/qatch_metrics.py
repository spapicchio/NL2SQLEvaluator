from collections import Counter
from enum import Enum
from itertools import chain
from typing import Callable, List, Tuple, Set

from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SQLExecutorOutput
from NL2SQLEvaluator.evaluator_nodes.evaluator_input import EvalText2SQLTask
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import register_node

logger = get_logger(__name__)


class QATCHMetric(Enum):
    """Enum for different metrics used in evaluation."""
    CELL_PRECISION = "cell_precision"
    CELL_RECALL = "cell_recall"
    TUPLE_CARDINALITY = "tuple_cardinality"
    TUPLE_CONSTRAINT = "tuple_constraint"
    TUPLE_ORDER = "tuple_order"
    CP_CR_TC = "cp_cr_tc"
    F1_SCORE = "f1_score"


@register_node(package_name='evaluator_nodes')
class QATCHEvaluator:
    """Evaluator for QATCH metrics with a dispatch-based architecture.

    Why: Separates the 'how to calculate' from the 'what to evaluate',
         making it easy to add metrics without touching the core loop.
    """

    def __init__(self):
        # Dispatch table mapping metrics to their implementation methods
        self._dispatch: dict[QATCHMetric, Callable] = {
            QATCHMetric.TUPLE_CARDINALITY: self._tuple_cardinality,
            QATCHMetric.TUPLE_CONSTRAINT: self._tuple_constraint,
            QATCHMetric.CELL_PRECISION: self._cell_precision,
            QATCHMetric.CELL_RECALL: self._cell_recall,
            QATCHMetric.TUPLE_ORDER: self._tuple_order,
            QATCHMetric.CP_CR_TC: self._cp_cr_tc,
            QATCHMetric.F1_SCORE: self._f1_score,
        }

    def execute_metric(self, tasks: list[EvalText2SQLTask], **kwargs) -> list[float]:
        if 'metric' not in kwargs:
            raise ValueError(f"Metric required. Choose from: {[m.value for m in QATCHMetric]}")

        metric_enum = QATCHMetric(kwargs.get("metric"))
        calc_method = self._dispatch.get(metric_enum)

        if not calc_method:
            raise NotImplementedError(f"Metric {metric_enum} logic not found."
                                      f"Available metrics: {[m.value for m in self._dispatch.keys()]}")

        results = []
        for task in tasks:
            if not task.predictions:
                logger.warning('No predictions provided, returning 0.0')
                results.append(0.0)
                continue

            # Majority voting happens once per task
            pred = task.get_best_pred_by_majority_voting(is_row_order_important=True)
            results.append(self._compute_score(pred, task.target, calc_method))

        return results

    def _compute_score(self, pred: SQLExecutorOutput, tar: SQLExecutorOutput,
                       metric_func: Callable) -> float:
        """Handles edge cases and rounding for all metrics.

        Why: Ensures consistent precision and behavior for empty sets
             without duplicating code in every sub-method.
        """
        len_p, len_t = len(pred), len(tar)

        if len_p == 0 and len_t == 0:
            return 1.0
        if (len_p == 0 and len_t != 0) or (len_p != 0 and len_t == 0):
            return 0.0

        raw_score = metric_func(pred, tar)
        return round(float(raw_score), 3)

    def _tuple_cardinality(self, pred: SQLExecutorOutput, tar: SQLExecutorOutput) -> float:
        return min(len(pred), len(tar)) / max(len(pred), len(tar))

    def _tuple_constraint(self, pred: SQLExecutorOutput, tar: SQLExecutorOutput) -> float:
        count_targ = Counter(tar.rows)
        count_pred = Counter(pred.rows)
        matches = [count_pred[k] == v for k, v in count_targ.items()]
        return sum(matches) / len(matches)

    def _cell_precision(self, pred: SQLExecutorOutput, tar: SQLExecutorOutput) -> float:
        p_cells, t_cells = self._get_cell_sets(pred, tar)
        return len(t_cells.intersection(p_cells)) / len(p_cells)

    def _cell_recall(self, pred: SQLExecutorOutput, tar: SQLExecutorOutput) -> float:
        p_cells, t_cells = self._get_cell_sets(pred, tar)
        return len(t_cells.intersection(p_cells)) / len(t_cells)

    def _tuple_order(self, pred: SQLExecutorOutput, tar: SQLExecutorOutput) -> float:
        # Filter rows to only common elements while preserving order
        p_common = [r for r in pred.rows if r in tar.rows]
        t_common = [r for r in tar.rows if r in pred.rows]

        if not t_common:
            return 0.0

        t_ranks = list(range(len(t_common)))
        # Create a lookup for target indices to avoid repeated .index() calls in a loop
        t_lookup = {row: i for i, row in enumerate(t_common)}
        p_ranks = [t_lookup[row] for row in p_common if row in t_lookup]

        # Calculate Spearman-like Rho
        n = max(len(t_common), 2)
        sum_sq_diff = sum((tr - pr) ** 2 for tr, pr in zip(t_ranks, p_ranks))
        rho = 1 - (6 * sum_sq_diff) / (n * (n ** 2 - 1))

        # Min-Max Normalization to [0, 1] range
        return (rho - (-1)) / (1 - (-1))

    def _f1_score(self, pred: SQLExecutorOutput, tar: SQLExecutorOutput) -> float:
        p = self._cell_precision(pred, tar)
        r = self._cell_recall(pred, tar)
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    def _cp_cr_tc(self, pred: SQLExecutorOutput, tar: SQLExecutorOutput) -> float:
        scores = [self._cell_precision(pred, tar),
                  self._cell_recall(pred, tar),
                  self._tuple_cardinality(pred, tar)]
        return sum(scores) / 3

    @staticmethod
    def _get_cell_sets(pred: SQLExecutorOutput, tar: SQLExecutorOutput) -> Tuple[Set, Set]:
        """Helper to flatten rows into unique cell sets."""
        return set(chain.from_iterable(pred.rows)), set(chain.from_iterable(tar.rows))
