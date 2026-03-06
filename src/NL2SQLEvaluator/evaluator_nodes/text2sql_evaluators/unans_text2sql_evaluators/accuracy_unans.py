from NL2SQLEvaluator.evaluator_nodes.evaluator_input import EvalUnansText2SQLTask
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import register_node

logger = get_logger(__name__)


@register_node(package_name='evaluator_nodes')
class UnansAccuracyEvaluator:
    def execute_metric(
            self,
            tasks: list[EvalUnansText2SQLTask],
            *args,
            **kwargs
    ) -> list[float]:
        #  Unanswerable accuracy metric introduced in EMNLP paper:
        #  "Squab: Evaluating LLM Robustness to Ambiguous and Unanswerable Questions in Semantic Parsing"
        #  https://underline.io/lecture/130658-squab-evaluating-llm-robustness-to-ambiguous-and-unanswerable-questions-in-semantic-parsing

        if 'unans_label' not in kwargs:
            raise ValueError("`unans_label` not provided in kwargs"
                             "This metric is a regex checks and needs to know what label to check for unanswerable questions.")

        unans_label = kwargs['unans_label']
        results = [self._ex(task, unans_label) for task in tasks]
        return results

    def _ex(self, task: EvalUnansText2SQLTask, unans_label: str) -> float:
        # check with regex if the unans_label is in the generated text
        if task.target and unans_label.lower() in task.predictions.lower():
            return 1.0
        elif task.target and unans_label.lower() not in task.predictions.lower():
            return 0.0
        elif not task.target and unans_label.lower() in task.predictions.lower():
            return 0.0
        elif not task.target and unans_label.lower() not in task.predictions.lower():
            return 1.0

        raise ValueError("Unexpected case in UnansAccuracyEvaluator._ex")
