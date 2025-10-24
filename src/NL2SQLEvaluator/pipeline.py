import pandas as pd
from pydantic import BaseModel, model_validator
from typing_extensions import Self

from NL2SQLEvaluator.config import ScriptArgs, DatasetArgs, ModelArgs, PipelineArgs
from NL2SQLEvaluator.dataset_reader_nodes.data_reader_protocol import ChatMessageHF
from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import OutputTable
from NL2SQLEvaluator.evaluator_nodes.evaluator_protocol import evaluate_target_and_pred
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import get_node_from_registry

logger = get_logger(__name__)


class PipelineInput(BaseModel):
    db_files: list[str]
    target_sql: list[list[str]]
    predictions: list[list[str]] | None = None
    input_seq: list[ChatMessageHF]
    executed_tar_sqls: list[list[OutputTable]] | None = None
    executed_pred_sqls: list[list[OutputTable]] | None = None
    scores: list[float] | None = None

    # validate target_sql, inputs seq and db_files all have same length
    @model_validator(mode='after')
    def check_len(self) -> Self:
        are_equal = len(self.db_files) == len(self.target_sql) == len(self.input_seq)
        if self.predictions is not None:
            are_equal = are_equal and (len(self.predictions) == len(self.db_files))
        if self.executed_tar_sqls is not None:
            are_equal = are_equal and (len(self.executed_tar_sqls) == len(self.db_files))
        if self.executed_pred_sqls is not None:
            are_equal = are_equal and (len(self.executed_pred_sqls) == len(self.db_files))
        if self.scores is not None:
            are_equal = are_equal and (len(self.scores) == len(self.db_files))

        if not are_equal:
            raise ValueError(
                "Length of db_files, target_sql, input_seq, predictions, executed_tar_sqls, and executed_pred_sqls must be equal."
            )
        return self

    def to_pandas(self):
        rows = []
        for i in range(len(self.db_files)):
            row = {
                "db_file": self.db_files[i],
                "input_seq": self.input_seq[i],
                "target_sql": self.target_sql[i],
                "predicted_sql": self.predictions[i] if self.predictions else None,
                "executed_target_sql": self.executed_tar_sqls[i] if self.executed_tar_sqls else None,
                "executed_predicted_sql": self.executed_pred_sqls[i] if self.executed_pred_sqls else None,
                "score": self.scores[i] if self.scores else None,
            }
            rows.append(row)

        return pd.DataFrame(rows)


def run_pipeline(data_input: PipelineInput, script_args: ScriptArgs, data_args: DatasetArgs, model_args: ModelArgs,
                 pipeline_args: PipelineArgs) -> PipelineInput:
    logger.info("Running evaluation with args:", (script_args, data_args, model_args, pipeline_args))
    predictor, cache_db, db_executor, evaluator = _get_pipeline_nodes(pipeline_args)
    data_with_predictions = _get_predictions(data_input, predictor, model_args)
    data_with_executed_sqls = _execute_sqls(data_with_predictions, db_executor, cache_db,
                                            script_args.cache_db_file_path)
    return _evaluate(data_with_executed_sqls, evaluator)


def _get_predictions(data_input: PipelineInput, predictor, model_args: ModelArgs) -> PipelineInput:
    if data_input.predictions is not None:
        logger.info("Predictions already exist in input, skipping prediction step.")
        return data_input

    from NL2SQLEvaluator.predictor_nodes.predictor_protocol import generate_predictions
    predictions = generate_predictions(
        predictor=predictor,
        model_name=model_args.model_name_or_path,
        multiple_tasks_messages=data_input.input_seq,
        model_args=model_args)

    return data_input.model_copy(update={"predictions": predictions})


def _execute_sqls(data_input: PipelineInput, db_executor, cache_db, cache_db_file) -> PipelineInput:
    from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import execute_queries_in_model_predictions
    to_be_executed = data_input.target_sql + data_input.predictions
    executed_queries = execute_queries_in_model_predictions(
        db_executor=db_executor,
        db_files=data_input.db_files,
        queries=to_be_executed,
        params=None,
        sql_cache_protocol=cache_db,
        cache_db_file=cache_db_file,
    )
    executed_tar_sqls = executed_queries[: len(data_input.target_sql)]
    executed_pred_sqls = executed_queries[len(data_input.target_sql):]

    return data_input.model_copy(update={
        "executed_tar_sqls": executed_tar_sqls,
        "executed_pred_sqls": executed_pred_sqls
    })


def _evaluate(data_input: PipelineInput, evaluator):
    evaluation_results = evaluate_target_and_pred(
        evaluator,
        multiple_tasks_preds=data_input.executed_pred_sqls,
        multiple_tasks_tars=data_input.executed_tar_sqls
    )
    return data_input.model_copy(update={"scores": evaluation_results})


def _get_pipeline_nodes(pipeline_args: PipelineArgs):
    predictor = get_node_from_registry('predictor_nodes', pipeline_args.predictor_node)
    cache_db = None
    if pipeline_args.sql_cache_node:
        cache_db = get_node_from_registry('db_executor_nodes', pipeline_args.sql_cache_node)
    db_executor = get_node_from_registry('db_executor_nodes', pipeline_args.db_executor_node)
    evaluator = get_node_from_registry('evaluator_nodes', pipeline_args.evaluator_node)
    return predictor, cache_db, db_executor, evaluator
