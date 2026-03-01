import dataclasses
import os
import statistics

from NL2SQLEvaluator.config import ScriptArgs, DatasetArgs, PipelineArgs, ModelArgs
from NL2SQLEvaluator.dataset_reader_nodes.data_reader_protocol import read_data_from_file, DataInput
from NL2SQLEvaluator.db_executor_nodes.utils import utils_extract_sql_or_same
from NL2SQLEvaluator.hf_parser import TrlParser
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import get_node_from_registry
from NL2SQLEvaluator.pipeline import PipelineInput, run_pipeline, tasks_to_pandas

logger = get_logger(__name__)


@dataclasses.dataclass
class SummaryResults:
    strategy: str
    model_name: str
    dataset_name: str
    metric: str
    value: float
    std: float


def run_evaluation(script_args: ScriptArgs, data_args: DatasetArgs, model_args: ModelArgs, pipeline_args: PipelineArgs):
    logger.info("Args: %s", (script_args, data_args, model_args, pipeline_args))

    if not data_args.dataset_path:
        raise ValueError("dataset_path must be set in DatasetArgs.")

    reader = get_node_from_registry('dataset_reader_nodes', pipeline_args.dataset_reader_node)
    dataset = read_data_from_file(reader, data_args.dataset_path, base_db_path=data_args.relative_db_base_path)
    if not dataset:
        logger.warning("Empty dataset: %s", data_args.dataset_path)
        return []

    num_experiments = script_args.num_of_experiments_to_get_std if model_args.temperature > 0 else 1
    logger.info('Running %d experiments to calculate standard deviation.', num_experiments)

    pipe_input = _prepare_input(dataset, data_args)
    # Replicate for multiple experiment runs (for std dev over stochastic sampling)
    pipe_input = PipelineInput(
        db_files=pipe_input.db_files * num_experiments,
        target_sql=pipe_input.target_sql * num_experiments,
        predictions=pipe_input.predictions * num_experiments if pipe_input.predictions else None,
        input_seq=pipe_input.input_seq * num_experiments if pipe_input.input_seq else None,
    )

    result = run_pipeline(pipe_input, script_args, model_args, pipeline_args)
    df = tasks_to_pandas(result.tasks)

    ex_n = []
    n = len(dataset)
    for i in range(num_experiments):
        start = i * n
        end = start + n
        n_pred = result.pred_sqls[start:end]
        n_scores = result.scores[start:end]
        df.loc[start:end - 1, f'{model_args.model_name_or_path}_{i}'] = [str(p) for p in n_pred]
        df.loc[start:end - 1, f'{pipeline_args.evaluator_node}_{i}'] = n_scores
        queries_i = [[utils_extract_sql_or_same(q) for q in query_list] for query_list in n_pred]
        valid_scores = [s for s in n_scores if s is not None]
        ex_n.append(statistics.mean(valid_scores) if valid_scores else 0.0)
        df.loc[start:end - 1, f'predicted_SQL_{i}'] = [str(q) for q in queries_i]
        df.loc[start:end - 1, f'EX_{i}'] = n_scores

    summary_results = SummaryResults(
        strategy='greedy' if model_args.number_of_completions == 1 else 'majority_voting',
        model_name=model_args.model_name_or_path,
        dataset_name=os.path.basename(data_args.dataset_path),
        metric=pipeline_args.evaluator_node,
        value=statistics.mean(ex_n) * 100,
        std=(statistics.stdev(ex_n) * 100) if len(ex_n) > 1 else 0.0
    )

    logger.warning(summary_results)

    if pipeline_args.saver_node is not None:
        saver = get_node_from_registry('saver_nodes', pipeline_args.saver_node)
        saver.save(script_args.output_dir, df=df,
                   configs=(script_args, data_args, model_args, pipeline_args, summary_results))

    return result.scores


def _prepare_input(dataset: list[DataInput], data_args: DatasetArgs) -> PipelineInput:
    """Convert validated DataInput records into a PipelineInput."""
    db_files = [item.db_id for item in dataset]
    target_sql = [item.target_code for item in dataset]

    predictions = None
    if data_args.pred_col_name:
        predictions = []
        for item in dataset:
            pred = item.model_extra.get(data_args.pred_col_name) if item.model_extra else None
            if isinstance(pred, str):
                pred = [pred]
            predictions.append(pred or [])

    input_seq = [item.input_seq for item in dataset]

    return PipelineInput(
        db_files=db_files,
        target_sql=target_sql,
        predictions=predictions,
        input_seq=input_seq,
    )


def main():
    parser = TrlParser(dataclass_types=[ScriptArgs, DatasetArgs, ModelArgs, PipelineArgs])
    script_args, data_args, model_args, pipeline_args = parser.parse_args_and_config()
    run_evaluation(script_args, data_args, model_args, pipeline_args)


if __name__ == "__main__":
    main()
