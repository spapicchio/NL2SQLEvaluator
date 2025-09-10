from dataclasses import dataclass, field
from typing import TypedDict

import pandas as pd
from dotenv import load_dotenv

from NL2SQLEvaluator.hf_argument_parser import TrlParser
from NL2SQLEvaluator.orchestrator import orchestrator_entrypoint
from NL2SQLEvaluator.orchestrator_state import MultipleTasks, flatten_multiple_tasks, AvailableMetrics, \
    AvailableDialect, SingleTask, SQLInstance
from NL2SQLEvaluator.utils import utils_read_dataset, utils_get_engine

load_dotenv(override=True)


class DatasetRow(TypedDict):
    target_query: str | list[str]
    predicted_query: str | list[str]
    db_id: str


@dataclass
class ScriptArgs:
    output_dir: str | None = field(
        default="./outputs",
        metadata={"help": "Directory where evaluation outputs will be saved"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"}
    )
    metrics: list[AvailableMetrics] = field(
        default_factory=lambda: [AvailableMetrics.EXECUTION_ACCURACY],
        metadata={"help": f"List of evaluation metrics to compute (e.g., {AvailableMetrics})"}
    )
    batch_size: int = field(
        default=50,
        metadata={"help": "Number of samples to process in each batch"}
    )

    ################
    # Dataset params
    ################
    relative_db_base_path: str = field(
        default="data/bird_dev/dev_databases",
        metadata={"help": "Relative path to the database files directory"}
    )

    database_dialect: AvailableDialect = field(
        default=AvailableDialect.sqlite,
        metadata={"help": f"Database dialect (e.g., {AvailableDialect})"}
    )

    dataset: list[DatasetRow] | None = field(
        default=None,
        metadata={
            "help": "The dataset already parsed in the correct format. If None the data is read from the dataset_path"
        }
    )

    dataset_path: str | None = field(
        default="simone-papicchio/bird",
        metadata={"help": "HuggingFace dataset path or local dataset path"}
    )
    dataset_name: str = field(
        default="bird-dev",
        metadata={"help": "Name of the dataset configuration to use"}
    )
    column_name_target: str | None = field(
        default="SQL",
        metadata={"help": "Column name in dataset for the target SQL queries"}
    )

    column_name_predicted: str | None = field(
        default="predicted_sql",
        metadata={"help": "Column name in dataset for the predicted SQL queries"}
    )

    column_name_db_id: str | None = field(
        default="db_id",
        metadata={"help": "Column name in dataset for the database identifiers"}
    )


def run_evaluation(
        script_args: ScriptArgs,
        **kwargs,
) -> tuple[dict, pd.DataFrame]:
    # read dataset

    dataset: list[dict] = script_args.dataset or utils_read_dataset(script_args.dataset_path)
    # crete the tasks for the orchestrator
    multiple_tasks = [
        utils_create_single_task_orchestration(row, script_args, **kwargs)
        for row in dataset
    ]
    multiple_tasks = MultipleTasks(tasks=multiple_tasks, batch_size=script_args.batch_size)
    # run the evaluation
    completed_tasks = orchestrator_entrypoint.invoke(multiple_tasks)

    # process the results in a DataFrame
    df_completed_tasks = pd.DataFrame(flatten_multiple_tasks(completed_tasks))
    # create summary with the metrics
    results = df_completed_tasks[[v.value for v in script_args.metrics]]
    summary = results.mean().to_dict()
    return summary, df_completed_tasks


def utils_create_single_task_orchestration(row: dict,
                                           script_args: ScriptArgs,
                                           **kwargs) -> SingleTask:
    target_sql = row.pop(script_args.column_name_target)
    predicted_sql = row.pop(script_args.column_name_predicted)
    db_id = row.pop(script_args.column_name_db_id)

    return SingleTask(
        relative_db_base_path=script_args.relative_db_base_path,
        dataset_name=script_args.dataset_name,
        dialect=script_args.database_dialect,
        engine=utils_get_engine(relative_base_path=script_args.relative_db_base_path,
                                db_executor=script_args.database_dialect, db_id=db_id),
        metrics=script_args.metrics,
        target_sql=SQLInstance(query=target_sql) if isinstance(target_sql, str) else [SQLInstance(query=sql)
                                                                                      for sql in target_sql],
        predicted_sql=SQLInstance(query=predicted_sql) if isinstance(predicted_sql, str) else [SQLInstance(query=sql)
                                                                                               for sql in
                                                                                               predicted_sql],
        db_id=db_id,
        external_metadata=kwargs if kwargs else None,
    )


if __name__ == "__main__":
    df = pd.read_json('data/bird_dev/dev.json')
    df['target_sql'] = df['SQL']
    df['predicted_sql'] = df['SQL']

    parser = TrlParser(ScriptArgs)
    script_args, config_remaining_strings = parser.parse_args_and_config()
    summary, df_eval = run_evaluation(script_args[0], **config_remaining_strings)
