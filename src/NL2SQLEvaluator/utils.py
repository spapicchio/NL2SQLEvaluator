import os
from functools import lru_cache

import pandas as pd
from datasets import load_dataset

from NL2SQLEvaluator.db_executor import BaseSQLDBExecutor
from NL2SQLEvaluator.orchestrator_state import AvailableDialect


@lru_cache(maxsize=100)
def utils_get_engine(relative_base_path, db_executor: AvailableDialect, db_id: str, *args, **kwargs) -> BaseSQLDBExecutor:
    try:
        if db_executor == AvailableDialect.sqlite:
            from NL2SQLEvaluator.db_executor.sqlite_executor import SqliteDBExecutor
            return SqliteDBExecutor.from_uri(
                relative_base_path=os.path.join(relative_base_path, db_id, f"{db_id}.sqlite"), *args, **kwargs)
    except Exception as e:
        raise ValueError(f"Error initializing database executor for {relative_base_path}: {e}")
    raise ValueError(f"Database executor not supported: {db_executor}. Supported: {list(AvailableDialect)}")


def utils_read_dataset(file_name) -> list[dict]:
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_name).to_list()

    elif file_name.endswith('.json'):
        df = pd.read_json(file_name)

    elif file_name.endswith('.parquet'):
        df = pd.read_parquet(file_name)

    else:
        # assume reading with Hugging Face datasets library
        dataset = load_dataset(file_name)
        dataset = dataset['dev'] if 'dev' in dataset else dataset[list(dataset.keys())[0]]
        df = dataset.to_pandas()

    # # TODO remove only used for debugging
    # df['predicted_sql'] = df['SQL']
    # df = df[:100]
    # # TODO ---------------------------

    return df.to_dict(orient='records')
