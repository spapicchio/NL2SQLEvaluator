import dataclasses
import os
import statistics

from NL2SQLEvaluator.config import ScriptArgs, DatasetArgs, PipelineArgs, ModelArgs
from NL2SQLEvaluator.dataset_reader_nodes.data_reader_protocol import read_data_from_file
from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import extract_sql_or_same
from NL2SQLEvaluator.hf_parser import TrlParser
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import get_node_from_registry
from NL2SQLEvaluator.pipeline import run_pipeline

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
    # read dataset:
    reader = get_node_from_registry('dataset_reader_nodes', pipeline_args.dataset_reader_node)
    dataset = read_data_from_file(reader, data_args.dataset_path, base_db_path=data_args.relative_db_base_path)
    dataset = dataset[:10]
    if not dataset:
        logger.warning("Empty dataset: %s", data_args.dataset_path)

    num_experiments = script_args.num_of_experiments_to_get_std if model_args.temperature > 0 else 1
    logger.info(f'Running {num_experiments} experiments to calculate standard deviation.')
    db_files, target_sqls, pred_sqls, input_seqs = _prepare_input(dataset, data_args)
    data_input = PipelineInput(
        db_files=db_files * num_experiments,
        target_sql=target_sqls * num_experiments,
        predictions=pred_sqls * num_experiments if data_args.pred_col_name else None,
        input_seq=input_seqs * num_experiments
    )

    data_with_scores = run_pipeline(data_input, script_args, data_args, model_args, pipeline_args)
    df = data_with_scores.to_pandas()

    ex_n = []
    for i in range(num_experiments):
        start = i * len(dataset)
        end = start + len(dataset)
        n_pred = data_with_scores.predictions[start:end]
        n_scores = data_with_scores.scores[start:end]
        df[f'{model_args.model_name_or_path}_{i}'] = n_pred
        df[f'{pipeline_args.evaluator_node}_{i}'] = n_scores
        queries_i = [[extract_sql_or_same(query) for query in query_list] for query_list in n_pred]
        ex_n.append(statistics.mean(n_scores))
        df[f'predicted_SQL_{i}'] = queries_i
        df[f'EX_{i}'] = n_scores

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

    return data_input.scores


def _prepare_input(dataset, data_args):
    db_files = []
    target_sqls = []
    pred_sqls = []
    input_seqs = []
    for item in dataset:
        db_files.append(item['db_file'])

        target_ = item[data_args.target_seq_col_name]
        if isinstance(target_, str):
            target_ = [target_]
        target_sqls.append(target_)

        pred_ = None
        if data_args.pred_col_name:
            pred_ = item[data_args.pred_col_name]
            if isinstance(pred_, str):
                pred_ = [pred_]

        pred_sqls.append(pred_)

        input_seqs.append(item[data_args.input_seq_col_name])

    return db_files, target_sqls, pred_sqls, input_seqs


def main():
    parser = TrlParser(dataclass_types=[ScriptArgs, DatasetArgs, ModelArgs, PipelineArgs])  # or [ScriptArgs]
    script_args, data_args, model_args, pipeline_args = parser.parse_args_and_config()
    run_evaluation(script_args, data_args, model_args, pipeline_args)


if __name__ == "__main__":
    main()
