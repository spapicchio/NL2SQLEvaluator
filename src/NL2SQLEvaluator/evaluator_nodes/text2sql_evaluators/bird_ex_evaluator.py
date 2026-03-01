from NL2SQLEvaluator.evaluator_nodes.evaluator_input import EvalText2SQLTask
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import register_node

logger = get_logger(__name__)


@register_node(package_name='evaluator_nodes')
class BirdEXEvaluator:
    """Implementation of the standard EX from BIRD: https://bird-bench.github.io/"""

    def execute_metric(
            self,
            tasks: list[EvalText2SQLTask],
            *args,
            **kwargs
    ) -> list[float]:
        results = [self._ex(task) for task in tasks]
        return results

    def _ex(self, task: EvalText2SQLTask) -> float:
        if len(task.predictions) == 0:
            logger.warning('No predictions provided for EX evaluation, returning 0.0')
            return 0.0

        target = frozenset(task.target.rows)
        majority_vote = task.get_best_pred_by_majority_voting(
            is_row_order_important=False,
            is_order_column=False
        )

        if len(majority_vote) == len(target) == 0:
            return 1.0
        elif len(majority_vote) != 0 and len(target) == 0 or len(majority_vote) == 0 and len(target) != 0:
            return 0.0

        return float(target == frozenset(tuple(majority_vote.rows)))
