from enum import Enum
from typing import Literal, Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, field_validator

from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SQLExecutorOutput, GenericExecutorOutput

TargetType = TypeVar("TargetType")
PredType = TypeVar("PredType")


class EvaluationType(str, Enum):
    """Enumeration of supported evaluation metrics and their specific logic."""
    TEXT2SQL = "text2sql"
    AMBIGTEXT2SQL = "ambig_text2sql"
    UNANSTEXT2SQL = "unans_text2sql"
    TEXT2CYPHER = "text2cypher"


class GenericOutputTable:
    pass


class BaseEvalTask(BaseModel, Generic[TargetType, PredType]):
    """Base schema for evaluation data."""
    model_config = ConfigDict(extra='allow')

    target: TargetType
    predictions: PredType

    @field_validator('target')
    @classmethod
    def target_must_not_be_empty(cls, v: Any) -> Any:
        """Ensures the target is not an empty collection."""
        # We only check length if the type is a collection (list, set, etc.)
        if hasattr(v, '__len__') and not isinstance(v, GenericExecutorOutput) and len(v) == 0:
            raise ValueError("Target cannot be empty.")
        return v


class EvalText2SQLTask(BaseEvalTask[SQLExecutorOutput, list[SQLExecutorOutput]]):
    """Specific task for Text2SQL tasks.
    Text2SQL tasks are a common evaluation scenario where the model generates SQL queries based on natural language input.
    Note:
        the target is a single SQLExecutorOutput representing the correct UNAMBIGUOUS SQL query,
        while the predictions are a list of SQLExecutorOutput objects representing the model's generated queries.
        if multiple predictions are provided, the evaluation logic will consider the best match among them.
    """
    task_type: Literal[EvaluationType.TEXT2SQL] = EvaluationType.TEXT2SQL

    def get_best_pred_by_majority_voting(self, is_row_order_important, is_order_column=True) -> SQLExecutorOutput:
        """Selects the prediction that appears most frequently based on result rows.

                Why: In Self-Consistency (CoT), we take multiple samples and find the consensus.
                How:
                    >>> task = EvalText2SQLTask(target=t, predictions=[p1, p2, p2])
                    >>> best = task.get_best_pred_by_majority_voting() # Returns p2

                Args:
                    is_row_order_important: If True, [1, 2] != [2, 1].

                Returns:
                    The SQLExecutorOutput object that represents the majority consensus.
                """

        def get_canonical_form(pred_: SQLExecutorOutput):
            sorted_pred_rows = tuple(map(pred_.sort_with_universal_key, pred_.rows))
            return tuple(sorted(sorted_pred_rows, key=SQLExecutorOutput.universal_sort_key)) if not is_row_order_important else sorted_pred_rows

        # the map contain the canonical form of the prediction as key and a list of [frequency, original_prediction] as value
        frequency_map: dict[Any, list] = dict()
        for pred in self.predictions:
            canon_rows = get_canonical_form(pred) if is_order_column else tuple(pred.rows)

            if canon_rows in frequency_map:
                frequency_map[canon_rows][0] += 1
            else:
                frequency_map[canon_rows] = [1, pred]

        # Find the entry with the highest frequency
        # In case of Tie, max returns the first one it encounters, which is fine since they are all tied.
        best_canonical_key = max(frequency_map, key=lambda k: frequency_map[k][0])
        return frequency_map[best_canonical_key][1]


class EvalAmbigText2SQLTask(EvalText2SQLTask):
    """Specific task for AMBIGUOUS Text2SQL tasks.
    Ambiguous Text2SQL tasks are a common evaluation scenario where the model generates SQL queries based on natural language input
    and the predicted queries are more than one.

    Note:
        the target is a list of SQLExecutorOutput representing all the possible SQL target queries for the ambiguous question.
        For each target query, the best matching prediction will be considered in the evaluation logic.
    """
    task_type: Literal[EvaluationType.AMBIGTEXT2SQL] = EvaluationType.AMBIGTEXT2SQL  # pyrefly: ignore
    target: list[SQLExecutorOutput]  # pyrefly: ignore


class EvalUnansText2SQLTask(BaseEvalTask[bool, str]):
    """Unanswerbale Text2SQL
    Target is boolean, wheter the question is unanswerable or not,
    Predictions is a string, which is the model's generated responses.
    """
    task_type: Literal[EvaluationType.UNANSTEXT2SQL] = EvaluationType.UNANSTEXT2SQL


class EvalText2CypherTask(BaseEvalTask[SQLExecutorOutput, list[SQLExecutorOutput]]):
    pass
