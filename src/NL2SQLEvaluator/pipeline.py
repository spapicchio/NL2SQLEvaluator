import dataclasses

import pandas as pd
from pydantic import BaseModel, model_validator, ConfigDict
from typing_extensions import Self

from NL2SQLEvaluator.config import ScriptArgs, ModelArgs, PipelineArgs
from NL2SQLEvaluator.dataset_reader_nodes.data_reader_protocol import ChatMessageHF
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SQLExecutorOutput, ExecutorError
from NL2SQLEvaluator.evaluator_nodes.evaluator_protocol import evaluate_target_and_pred
from NL2SQLEvaluator.evaluator_nodes.evaluator_input import EvaluationType
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import get_node_from_registry

logger = get_logger(__name__)


# ---- Entry point for callers that bypass the dataset reader ----

class PipelineInput(BaseModel):
    """Flat multi-item container used as the entry point to run_pipeline.

    Fields are parallel lists: index i across all lists describes task i.
    At least one of `predictions` or `input_seq` must be provided.
    """
    db_files: list[str]
    target_sql: list[list[str]]
    predictions: list[list[str]] | None = None
    input_seq: list[ChatMessageHF] | None = None

    @model_validator(mode='after')
    def check_lengths(self) -> Self:
        n = len(self.db_files)
        if self.predictions is None and self.input_seq is None:
            raise ValueError("Must provide at least one of `predictions` or `input_seq`.")
        if len(self.target_sql) != n:
            raise ValueError("target_sql length must match db_files length.")
        if self.predictions is not None and len(self.predictions) != n:
            raise ValueError("predictions length must match db_files length.")
        if self.input_seq is not None and len(self.input_seq) != n:
            raise ValueError("input_seq length must match db_files length.")
        return self


# ---- Per-task state accumulator (internal to the pipeline stages) ----

class PipelineTask(BaseModel):
    """Holds the evolving state of a single evaluation task as it moves through pipeline stages.

    Each stage populates the next field:
      reader       → db_path, target_sqls, input_seq
      predictor    → pred_sqls
      db_executor  → executed_targets, executed_preds
      evaluator    → score
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    db_path: str
    target_sqls: list[str]
    input_seq: ChatMessageHF | None = None
    pred_sqls: list[str] | None = None
    executed_targets: list[SQLExecutorOutput | ExecutorError] | None = None
    executed_preds: list[SQLExecutorOutput | ExecutorError] | None = None
    score: float | None = None

    @model_validator(mode='after')
    def check_has_input(self) -> Self:
        if self.pred_sqls is None and self.input_seq is None:
            raise ValueError("PipelineTask must have at least one of `pred_sqls` or `input_seq`.")
        return self


# ---- Pipeline output ----

@dataclasses.dataclass
class PipelineResult:
    """Collected outputs after all pipeline stages have run."""
    scores: list[float | None]
    pred_sqls: list[list[str]]
    tasks: list[PipelineTask]


# ---- Helpers ----

def _pipe_input_to_tasks(pipe_input: PipelineInput) -> list[PipelineTask]:
    """Convert the flat PipelineInput into one PipelineTask per row."""
    tasks = []
    for i, db_path in enumerate(pipe_input.db_files):
        tasks.append(PipelineTask(
            db_path=db_path,
            target_sqls=pipe_input.target_sql[i],
            input_seq=pipe_input.input_seq[i] if pipe_input.input_seq else None,
            pred_sqls=pipe_input.predictions[i] if pipe_input.predictions else None,
        ))
    return tasks


def tasks_to_pandas(tasks: list[PipelineTask]) -> pd.DataFrame:
    rows = []
    for task in tasks:
        rows.append({
            "db_path": task.db_path,
            "target_sqls": task.target_sqls,
            "input_seq": task.input_seq,
            "pred_sqls": task.pred_sqls,
            "executed_targets": task.executed_targets,
            "executed_preds": task.executed_preds,
            "score": task.score,
        })
    return pd.DataFrame(rows)


def _to_pipeline_result(tasks: list[PipelineTask]) -> PipelineResult:
    return PipelineResult(
        scores=[t.score for t in tasks],
        pred_sqls=[t.pred_sqls or [] for t in tasks],
        tasks=tasks,
    )


# ---- Public entry point ----

def run_pipeline(
        data_input: PipelineInput,
        script_args: ScriptArgs = ScriptArgs(),
        model_args: ModelArgs = ModelArgs(),
        pipeline_args: PipelineArgs = PipelineArgs(),
) -> PipelineResult:
    logger.info("Running pipeline with args: %s", (script_args, model_args, pipeline_args))
    predictor, cache_db, db_executor, evaluator = _get_pipeline_nodes(pipeline_args, script_args.cache_db_file_path)

    tasks = _pipe_input_to_tasks(data_input)
    tasks = _get_predictions(tasks, predictor, model_args)
    tasks = _execute_sqls(tasks, db_executor, cache_db, script_args.execution_timeout)
    tasks = _evaluate(tasks, evaluator, pipeline_args.evaluation_type)
    return _to_pipeline_result(tasks)


# ---- Stage functions (each operates on list[PipelineTask]) ----

def _get_predictions(tasks: list[PipelineTask], predictor, model_args: ModelArgs) -> list[PipelineTask]:
    """Predictor stage: populate pred_sqls from input_seq via the LLM."""
    if tasks[0].input_seq is None:
        return tasks  # predictions already pre-loaded; input_seq not needed

    from NL2SQLEvaluator.predictor_nodes.predictor_protocol import generate_predictions
    # All tasks have input_seq at this point (checked above via tasks[0])
    input_seqs: list[ChatMessageHF] = [t.input_seq for t in tasks]  # type: ignore[misc]
    predictions = generate_predictions(
        predictor=predictor,
        model_name=model_args.model_name_or_path,
        multiple_tasks_messages=input_seqs,
        model_args=model_args,
    )
    return [task.model_copy(update={"pred_sqls": predictions[i]}) for i, task in enumerate(tasks)]


def _execute_sqls(
        tasks: list[PipelineTask],
        db_executor,
        cache_db,
        timeout: int,
) -> list[PipelineTask]:
    """DB executor stage: run target and predicted SQL for each task."""
    from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import execute_queries_in_model_predictions

    db_paths = [t.db_path for t in tasks]
    shared_kwargs = dict(
        code_executor=db_executor,
        cached_db=cache_db,
        timeout=timeout,
    )

    # Execute gold targets
    target_results = execute_queries_in_model_predictions(
        db_paths=db_paths,
        queries=[t.target_sqls for t in tasks],
        **shared_kwargs,
    )

    # Execute predictions (skip if none provided)
    has_preds = any(t.pred_sqls for t in tasks)
    pred_results: list = [None] * len(tasks)
    if has_preds:
        pred_results = execute_queries_in_model_predictions(
            db_paths=db_paths,
            queries=[t.pred_sqls or [] for t in tasks],
            **shared_kwargs,
        )

    return [
        task.model_copy(update={
            "executed_targets": target_results[i],
            "executed_preds": pred_results[i],
        })
        for i, task in enumerate(tasks)
    ]


def _evaluate(tasks: list[PipelineTask], evaluator, evaluation_type: str) -> list[PipelineTask]:
    """Evaluator stage: score each task and populate task.score."""
    task_type = EvaluationType(evaluation_type)

    # For TEXT2SQL: target is the single gold execution result; predictions is all pred results.
    # executed_targets/preds are guaranteed non-None here (set by _execute_sqls).
    targets = []
    predictions = []
    for task in tasks:
        if task.executed_targets is None:
            raise ValueError(f"Task for db '{task.db_path}' has no executed_targets. Run _execute_sqls first.")
        targets.append(task.executed_targets[0])
        predictions.append(task.executed_preds)

    scores = evaluate_target_and_pred(
        evaluator,
        targets=targets,  # type: ignore[arg-type]
        predictions=predictions,  # type: ignore[arg-type]
        task_type=task_type,
    )
    return [task.model_copy(update={"score": scores[i]}) for i, task in enumerate(tasks)]


def _get_pipeline_nodes(pipeline_args: PipelineArgs, cache_db_file: str | None):
    predictor = None
    if pipeline_args.predictor_node is not None:
        predictor = get_node_from_registry('predictor_nodes', pipeline_args.predictor_node)()
    cache_db = None
    if pipeline_args.sql_cache_node and cache_db_file:
        cache_db = get_node_from_registry('db_executor_nodes', pipeline_args.sql_cache_node)(cache_db_file)
    db_executor = get_node_from_registry('db_executor_nodes', pipeline_args.db_executor_node)()
    evaluator = get_node_from_registry('evaluator_nodes', pipeline_args.evaluator_node)()
    return predictor, cache_db, db_executor, evaluator
