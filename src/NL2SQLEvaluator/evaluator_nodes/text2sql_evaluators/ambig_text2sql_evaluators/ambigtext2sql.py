from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SQLExecutorOutput
from NL2SQLEvaluator.evaluator_nodes.evaluator_input import EvalAmbigText2SQLTask, EvalText2SQLTask
from NL2SQLEvaluator.evaluator_nodes.text2sql_evaluators.ex_with_card import EXEvaluator
from NL2SQLEvaluator.evaluator_nodes.text2sql_evaluators.qatch_metrics import QATCHEvaluator
from NL2SQLEvaluator.node_registry import register_node


@register_node(package_name='evaluator_nodes')
class AmbigText2SQLEvaluator:
    """Evaluator for Ambiguous Text2SQL tasks using bipartite matching."""

    def __init__(self):
        # Cache evaluators to avoid re-instantiation overhead
        self._evaluators = {
            'qatch': QATCHEvaluator(),
            'ex': EXEvaluator()
        }

    def execute_metric(self, tasks: list[EvalAmbigText2SQLTask], **kwargs) -> list[float]:
        # Cleanly extract parameters with defaults
        params = {
            "metric": kwargs.get('metric', 'precision'),
            "evaluate_with": kwargs.get('evaluate_with', 'ex'),
            "is_row_order_important": kwargs.get('is_row_order_important', False),
            "threshold": kwargs.get('threshold', 0.6)  # Make threshold configurable
        }

        return [self._calculate_metric(task, **params) for task in tasks]

    def _calculate_metric(self, task: EvalAmbigText2SQLTask, **params) -> float:
        if not task.predictions:
            return 0.0

        predictions2match, target2match = self._calculate_combinations(task, **params)

        metric = params['metric']
        if metric == 'precision':
            return sum(predictions2match.values()) / len(predictions2match)
        if metric == 'recall':
            return sum(target2match.values()) / len(target2match)
        if metric == 'f1':
            p = sum(predictions2match.values()) / len(predictions2match)
            r = sum(target2match.values()) / len(target2match)
            return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        raise ValueError(f"Unsupported metric: {metric}")

    def _calculate_combinations(self, task: EvalAmbigText2SQLTask, **params):
        """Perform bipartite matching between predictions and targets."""
        evaluator = self._evaluators.get(params['evaluate_with'], self._evaluators['ex'])

        # Helper to create a stable key based on row order importance
        def get_key(output: SQLExecutorOutput):
            rows = output.rows
            if not params['is_row_order_important']:
                # Sort rows to ensure [(1,2)] and [(2,1)] produce the same key
                rows = sorted(rows, key=SQLExecutorOutput.universal_sort_key)
            return tuple(rows)

        # Initialize match tracking using canonical keys
        canonical_pred_rows2pred = {get_key(p): p for p in task.predictions}
        canonical_tarrows2target = {get_key(t): t for t in task.target}

        # initialize match tracking
        predictions_matched = {k: False for k in canonical_pred_rows2pred}
        targets_matched = {k: False for k in canonical_tarrows2target}

        # Compare every unique prediction against every unique target
        for p_key, pred in canonical_pred_rows2pred.items():
            for t_key, tar in canonical_tarrows2target.items():
                # If target already matched by someone else, we still check
                # to mark THIS prediction as correct.
                score = evaluator.execute_metric(
                    tasks=[EvalText2SQLTask(target=tar, predictions=[pred])],
                    is_row_order_important=params['is_row_order_important'],
                    metric='cp_cr_tc'
                )[0]

                if score >= params['threshold']:
                    predictions_matched[p_key] = True
                    targets_matched[t_key] = True

        return predictions_matched, targets_matched
