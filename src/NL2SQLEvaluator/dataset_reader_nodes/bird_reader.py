import pandas as pd

from NL2SQLEvaluator.dataset_reader_nodes.data_reader_protocol import DataInput, enforce_input_type
from NL2SQLEvaluator.node_registry import register_node


@register_node()
class ReadBird:
    @staticmethod
    def read(file_path: str, base_db_path: str, *args, **kwargs) -> list[DataInput]:
        if file_path.endswith(".json"):
            df = pd.read_json(file_path)
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            import datasets
            df = datasets.load_dataset(file_path).to_pandas()

        df['db_file'] = df['db_id'].map(lambda row: f"{base_db_path}/{row}/{row}.sqlite")
        df['target_query'] = df['SQL']
        values = df.to_dict(orient='records')
        enforce_input_type(values)
        return values
