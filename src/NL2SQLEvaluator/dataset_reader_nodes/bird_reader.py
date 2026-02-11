"""BIRD Dataset Reader implementation.

Why:
    Specifically handles the BIRD dataset format, mapping its internal 
    schema (db_id, SQL, etc.) to the standardized DataInput format used 
    by the NL2SQL evaluator.

How:
    >>> reader = ReadBirdData()
    >>> data = read_data_from_file(reader, "bird_train.json", base_db_path="./dbs")
"""

from pathlib import Path
from typing import Any

import pandas as pd

from NL2SQLEvaluator.node_registry import register_node


@register_node(package_name="dataset_readers")
class ReadBirdData:
    """Reader for the BIRD dataset supporting JSON, CSV, and Hugging Face Hub."""

    def read(self, file_path: str, base_db_path: str, **kwargs: Any) -> list[dict]:
        """Reads BIRD data and transforms it to a compatible dictionary format.

        Args:
            file_path: Local path or Hugging Face dataset ID.
            base_db_path: The root directory where .sqlite files are stored.
            **kwargs: Additional arguments like 'split' for HF datasets.

        Returns:
            list[dict]: A list of raw dictionaries matching the DataInput schema.
        """
        if file_path.endswith(".json"):
            df = pd.read_json(file_path)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            import datasets
            split = kwargs.get("split", "train")
            ds = datasets.load_dataset(file_path, split=split)
            df = ds.to_pandas()

        # Map BIRD specific columns to our standard DataInput schema
        # BIRD uses 'db_id' for the folder name and 'question' for the input
        df['db_id'] = df['db_id'].apply(lambda x: str(Path(base_db_path) / x / f"{x}.sqlite"))

        # In BIRD, 'SQL' is the target. We wrap it in a list as per DataInput
        df['target_code'] = df['SQL'].apply(lambda x: [x] if isinstance(x, str) else x)

        # Construct the conversation sequence
        df['input_seq'] = df['question'].apply(
            lambda q: [{"role": "user", "content": q}]
        )

        # Convert to list of dicts for Pydantic validation in the Orchestrator
        return df.to_dict(orient='records')
