"""SPARQL Dataset Reader implementation.

Why:
    Handles SPARQL dataset formats, mapping its internal schema
    (SPARQL, question, etc.) to the standardized DataInput format
    used by the NL2SQL evaluator.

How:
    >>> reader = ReadSparqlData()
    >>> data = read_data_from_file(reader, "sparql_data.json", sparql_endpoint_url="http://localhost:8890/sparql")
"""

from typing import Any

import pandas as pd

from NL2SQLEvaluator.node_registry import register_node


@register_node(package_name="dataset_readers")
class ReadSparqlData:
    """Reader for SPARQL datasets supporting JSON, CSV, and Hugging Face Hub."""

    def read(self, file_path: str, sparql_endpoint_url: str, **kwargs: Any) -> list[dict]:
        """Reads SPARQL data and transforms it to a compatible dictionary format.

        Expected columns in the dataset:
            - ``question``: natural language question
            - ``SPARQL``: gold SPARQL query

        Args:
            file_path: Local path or Hugging Face dataset ID.
            sparql_endpoint_url: The SPARQL endpoint URL to execute queries against.
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
            split = kwargs.get("split", "test")
            ds = datasets.load_dataset(file_path, split=split)
            df = ds.to_pandas()

        # All records target the same SPARQL endpoint
        df['db_id'] = sparql_endpoint_url

        # 'SPARQL' is the target. We wrap it in a list as per DataInput
        df['target_code'] = df['SPARQL'].apply(lambda x: [x] if isinstance(x, str) else x)

        # Construct the conversation sequence
        df['input_seq'] = df['question'].apply(
            lambda q: [{"role": "user", "content": q}]
        )

        return df.to_dict(orient='records')
