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

    # Optional: accept case-insensitive and aliases like "val" / "validation"
    _ALIASES = {
        'execution_accuracy': 'EXECUTION_ACCURACY',
        'f1_score': 'F1_SCORE',
        'cell_precision': 'CELL_PRECISION',
        'cell_recall': 'CELL_RECALL',
        'tuple_cardinality': 'TUPLE_CARDINALITY',
        'tuple_constraint': 'TUPLE_CONSTRAINT',
        'tuple_order': 'TUPLE_ORDER',
    }

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            v = value.strip().lower()
            v = cls._ALIASES.get(v, v)
            for m in cls:
                if m.value == v or m.name == v:
                    return m
        return None


class SQLInstance(BaseModel):
    query: str | None = None
    executed: list[tuple] | None = None


class SingleTask(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    engine: Any | None = None

    db_id: str
    dialect: AvailableDialect
    relative_db_base_path: str
    dataset_name: str
    metrics: list[AvailableMetrics] = [AvailableMetrics.EXECUTION_ACCURACY]
    target_sql: SQLInstance | list[SQLInstance]
    predicted_sql: SQLInstance | list[SQLInstance] | None = None
    results: dict[AvailableMetrics, float] | None = None

    external_metadata: dict[str, Any] | None = None


class MultipleTasks(BaseModel):
    tasks: list[SingleTask]
    batch_size: int = 50


def flatten_multiple_tasks(multiple_tasks: MultipleTasks) -> list:
    multiple_tasks = multiple_tasks.model_dump()
    tasks = multiple_tasks.pop("tasks")
    rows = []
    for task in tasks:
        task['dialect'] = task['dialect'].value
        # remove engine
        task.pop('engine')
        task.pop('metrics')
        results = {result.value: val for result, val in task.pop('results').items()}
        task = task | results
        task['target_sql'] = task["target_sql"]['query']
        task['predicted_sql'] = task["predicted_sql"]['query']
        task = task | task.pop('external_metadata')
        rows.append(task)

    return rows
