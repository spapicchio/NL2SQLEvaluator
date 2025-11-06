import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import register_node

logger = get_logger(__name__)


@register_node()
class JSONSaver:
    @staticmethod
    def save(folder: Path, df: pd.DataFrame, configs: tuple[Any], *args, **kwargs):
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        hours = datetime.now().strftime("%H-%M-%S")
        path = Path(folder) / date_str / hours
        path.mkdir(parents=True, exist_ok=True)

        df_path = path / 'df.json'
        df.to_json(df_path, orient='records', indent=4)
        logger.info(f"Saved DF in {df_path} as JSON")

        for config in configs:
            config_name = config.__class__.__name__ if is_dataclass(config) else type(config).__name__
            config_path = path / f"{config_name}.json"

            config_data = asdict(config) if is_dataclass(config) else config
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=4, default=str)
            logger.info(f"Saved config in {config_path} as JSON")
