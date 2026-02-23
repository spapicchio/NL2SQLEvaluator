"""Cypher Dataset Reader implementation.

Why:
    Handles Cypher dataset formats, mapping its internal schema
    (db_id, Cypher, question) to the standardized DataInput format
    used by the NL2SQL evaluator. The db_id column contains graph
    names that are resolved to Neo4j Bolt URIs via neo4j_info.json.

How:
    >>> reader = ReadCypherData()
    >>> data = read_data_from_file(reader, "cypher_data.json", neo4j_info_path="neo4j_info.json")
"""

import json
from typing import Any

import pandas as pd

from NL2SQLEvaluator.node_registry import register_node


@register_node(package_name="dataset_readers")
class ReadCypherData:
    """Reader for Cypher datasets supporting JSON, CSV, and Hugging Face Hub."""

    def read(self, file_path: str, neo4j_info_path: str, **kwargs: Any) -> list[dict]:
        """Reads Cypher data and transforms it to a compatible dictionary format.

        Expected columns in the dataset:
            - ``question``: natural language question
            - ``Cypher``: gold Cypher query
            - ``db_id``: graph name, resolved to a Bolt URI via neo4j_info.json

        Args:
            file_path: Local path or Hugging Face dataset ID.
            neo4j_info_path: Path to neo4j_info.json containing graph → connection mapping.
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

        graph_to_bolt = self._build_bolt_mapping(neo4j_info_path, **kwargs)

        # Map graph name → bolt URI
        def _resolve_bolt(graph_name: str) -> str:
            if graph_name not in graph_to_bolt:
                raise KeyError(
                    f"Graph '{graph_name}' not found in neo4j_info. "
                    f"Available graphs: {sorted(graph_to_bolt.keys())}"
                )
            return graph_to_bolt[graph_name]

        df['db_id'] = df['db_id'].apply(_resolve_bolt)

        # 'Cypher' is the target. We wrap it in a list as per DataInput
        df['target_code'] = df['Cypher'].apply(lambda x: [x] if isinstance(x, str) else x)

        # Construct the conversation sequence
        df['input_seq'] = df['question'].apply(
            lambda q: [{"role": "user", "content": q}]
        )

        return df.to_dict(orient='records')

    @staticmethod
    def _build_bolt_mapping(neo4j_info_path: str, **kwargs: Any) -> dict[str, str]:
        """Reads neo4j_info.json and builds a graph → bolt URI mapping.

        The JSON is expected to have the structure:
            { "<subset>": { "<graph>": {"host": ..., "port": ..., "username": ..., "password": ...} } }

        Args:
            neo4j_info_path: Path to the neo4j_info.json file.
            **kwargs: May include 'neo4j_info_subset' to select which subset (default "sampled").

        Returns:
            dict mapping graph name to bolt:// URI.
        """
        with open(neo4j_info_path) as f:
            neo4j_info = json.load(f)

        subset = kwargs.get("neo4j_info_subset", "sampled")
        graphs = neo4j_info[subset]

        mapping = {}
        for graph_name, conn in graphs.items():
            user = conn["username"]
            password = conn["password"]
            host = conn["host"]
            port = conn["port"]
            mapping[graph_name] = f"bolt://{user}:{password}@{host}:{port}"

        return mapping
