import pandas as pd
from pydantic import BaseModel, model_validator, ConfigDict
from typing_extensions import Self

from NL2SQLEvaluator.config import ScriptArgs, DatasetArgs, ModelArgs, PipelineArgs
from NL2SQLEvaluator.dataset_reader_nodes.data_reader_protocol import ChatMessageHF
from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import OutputTable
from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecutorError
from NL2SQLEvaluator.evaluator_nodes.evaluator_protocol import evaluate_target_and_pred
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import get_node_from_registry

logger = get_logger(__name__)


class PipelineTask(BaseModel):
    db_file: str
    target_sql: list[str]
    predictions: list[str] | None = None
    input_seq: ChatMessageHF | None = None
    executed_tar_sqls: list[OutputTable | ExecutorError] | None = None
    executed_pred_sqls: list[OutputTable | ExecutorError] | None = None
    score: float | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # validate target_sql, inputs seq and db_files all have same length
    @model_validator(mode='after')
    def check_len(self) -> Self:
        are_equal = True
        if self.predictions is None and self.input_seq is None:
            raise ValueError("Must provide at least input_seq or predictions.")

        if self.predictions is not None:
            are_equal = are_equal and (len(self.predictions) == len(self.target_sql))
        if self.executed_tar_sqls is not None:
            are_equal = are_equal and (len(self.executed_tar_sqls) == len(self.target_sql))
        if self.executed_pred_sqls is not None:
            are_equal = are_equal and (len(self.executed_pred_sqls) == len(self.target_sql))

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


def run_pipeline(pipeline_tasks=list[PipelineTask],
                 script_args: ScriptArgs = ScriptArgs(),
                 data_args: DatasetArgs = DatasetArgs(),
                 model_args: ModelArgs = ModelArgs(),
                 pipeline_args: PipelineArgs = PipelineArgs()) -> list[PipelineTask]:
    logger.info("Running evaluation with args:", (script_args, data_args, model_args, pipeline_args))
    predictor, cache_db, db_executor, evaluator = _get_pipeline_nodes(pipeline_args)

    data_with_predictions = _get_predictions(pipeline_tasks, predictor, model_args)
    data_with_executed_sqls = _execute_sqls(data_with_predictions, db_executor, cache_db,
                                            script_args.cache_db_file_path,
                                            script_args.execution_timeout)
    return _evaluate(data_with_executed_sqls, evaluator)


def _get_predictions(pipeline_tasks: list[PipelineTask], predictor, model_args: ModelArgs) -> list[PipelineTask]:
    tasks_input_seq = [task.input_seq for task in pipeline_tasks]
    if tasks_input_seq[0] is None:
        return pipeline_tasks

    from NL2SQLEvaluator.predictor_nodes.predictor_protocol import generate_predictions
    predictions = generate_predictions(
        predictor=predictor,
        model_name=model_args.model_name_or_path,
        multiple_tasks_messages=tasks_input_seq,
        model_args=model_args)

    return [task.model_copy(update={"predictions": predictions}) for task in pipeline_tasks]


def _execute_sqls(pipeline_tasks: list[PipelineTask], db_executor, cache_db, cache_db_file, timeout) -> list[
    PipelineTask]:
    from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import execute_queries_in_model_predictions
    to_be_executed = []
    [to_be_executed.append(task.target_sql) for task in pipeline_tasks]
    [to_be_executed.append(task.predictions) for task in pipeline_tasks]
    db_files = [task.db_file for task in pipeline_tasks] * 2

    executed_queries = execute_queries_in_model_predictions(
        db_executor=db_executor,
        db_files=db_files,
        queries=to_be_executed,
        params=None,
        sql_cache_protocol=cache_db,
        cache_db_file=cache_db_file,
        timeout=timeout,
    )
    executed_tar_sqls = executed_queries[: len(pipeline_tasks)]
    executed_pred_sqls = executed_queries[len(pipeline_tasks):]

    return [
        task.model_copy(update={"executed_tar_sqls": executed_tar_sqls, "executed_pred_sqls": executed_pred_sqls})
        for task in pipeline_tasks
    ]


def _evaluate(pipeline_tasks: list[PipelineTask], evaluator) -> list[PipelineTask]:
    executed_pred = []
    executed_target = []
    for task in pipeline_tasks:
        executed_pred.append(task.executed_pred_sqls)
        executed_target.append(task.executed_tar_sqls)

    evaluation_results = evaluate_target_and_pred(
        evaluator,
        multiple_tasks_preds=executed_pred,
        multiple_tasks_tars=executed_target
    )
    return [task.model_copy(update={"score": evaluation_results}) for task in pipeline_tasks]


def _get_pipeline_nodes(pipeline_args: PipelineArgs):
    predictor = None
    if pipeline_args.predictor_node is not None:
        predictor = get_node_from_registry('predictor_nodes', pipeline_args.predictor_node)
    cache_db = None
    if pipeline_args.sql_cache_node:
        cache_db = get_node_from_registry('db_executor_nodes', pipeline_args.sql_cache_node)

    db_executor = get_node_from_registry('db_executor_nodes', pipeline_args.db_executor_node)
    evaluator = get_node_from_registry('evaluator_nodes', pipeline_args.evaluator_node)
    return predictor, cache_db, db_executor, evaluator
