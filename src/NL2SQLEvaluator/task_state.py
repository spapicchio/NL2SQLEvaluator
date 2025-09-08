from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class AvailableDialect(Enum):
    sqlite = "sqlite"


class AvailableMetrics(Enum):
    """Enum for different metrics used in evaluation."""
    EXECUTION_ACCURACY = "execution_accuracy"
    F1_SCORE = "f1_score"
    CELL_PRECISION = "cell_precision"
    CELL_RECALL = "cell_recall"
    TUPLE_CARDINALITY = "tuple_cardinality"
    TUPLE_CONSTRAINT = "tuple_constraint"
    TUPLE_ORDER = "tuple_order"


class DataCfg(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dataset: str
    relative_db_base_path: str
    split: str = "val"
    dialect: AvailableDialect
    engine: Any | None = None


class EvalCfg(BaseModel):
    metrics: list[AvailableMetrics] = [AvailableMetrics.EXECUTION_ACCURACY]


class SQLInstance(BaseModel):
    query: str | None = None
    executed: list[tuple] | None = None


class SingleTask(BaseModel):
    dataset_parameters: DataCfg
    eval_parameters: EvalCfg
    target_sql: SQLInstance | list[SQLInstance]
    predicted_sql: SQLInstance | list[SQLInstance] | None = None
    results: dict[AvailableMetrics, float] | None = None
    external_metadata: dict[str, Any] | None = None


class MultipleTasks(BaseModel):
    tasks: list[SingleTask]
    batch_size: int = 50
