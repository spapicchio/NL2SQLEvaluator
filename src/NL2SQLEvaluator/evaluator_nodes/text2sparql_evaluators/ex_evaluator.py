from NL2SQLEvaluator.evaluator_nodes.evaluator_input import EvalText2SparqlTask
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import register_node

logger = get_logger(__name__)


@register_node(package_name='text2sparql_evaluators')
class SparqlEXEvaluator:
    """Execution accuracy evaluator for Text2SPARQL tasks.

    Delegates comparison to ``SparqlExecutorOutput.is_equivalent_to``,
    which handles column-permutation backtracking automatically.
    """

    def execute_metric(
            self,
            tasks: list[EvalText2SparqlTask],
            *args,
            **kwargs
    ) -> list[float]:
        is_row_order_important = kwargs.get('is_row_order_important', False)
        results = [self._ex(task, is_row_order_important) for task in tasks]
        return results

    def _ex(self, task: EvalText2SparqlTask, is_row_order_important: bool) -> float:
        if len(task.predictions) == 0:
            logger.warning('No predictions provided for EX evaluation, returning 0.0')
            return 0.0

        target = task.target
        majority_vote = task.get_best_pred_by_majority_voting(is_row_order_important=is_row_order_important)

        if len(majority_vote) == len(target) == 0:
            return 1.0
        elif len(majority_vote) != 0 and len(target) == 0 or len(majority_vote) == 0 and len(target) != 0:
            return 0.0

        return float(target.is_equivalent_to(majority_vote, is_row_order_important=is_row_order_important))
