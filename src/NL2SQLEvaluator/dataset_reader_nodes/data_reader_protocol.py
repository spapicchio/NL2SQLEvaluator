"""Data Ingestion Orchestrator for NL2SQL Evaluation.

Why:
    Acts as the single entry point for data loading. It enforces
    the DataInput schema at runtime, providing clear error reporting
    when dataset files deviate from the expected format.

How:
    >>> reader = MyCustomReader() # Implements DataReaderProtocol
    >>> try:
    >>>     dataset = read_data_from_file(reader, "path/to/data.json")
    >>> except ValidationError:
    >>>     # Handle invalid data
"""

from typing import Protocol, Literal, Any, List, runtime_checkable

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from NL2SQLEvaluator.logger import get_logger

logger = get_logger(__name__)


# ---- Schema Definitions ----

class ChatTurn(BaseModel):
    """Schema for a single conversation turn."""
    model_config = ConfigDict(extra='allow')
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str


# A conversation is a sequence of turns (matches HuggingFace chat format)
ChatMessageHF = list[ChatTurn]


class DataInput(BaseModel):
    """The mandatory schema for all evaluator inputs."""
    model_config = ConfigDict(extra='allow')
    db_id: str
    target_code: List[str]
    input_seq: List[ChatTurn]


@runtime_checkable
class DataReadProcessProtocol(Protocol):
    """Interface for components that load raw data into memory."""

    def read(self, file_path: str, **kwargs) -> List[dict]:
        """Reads a file and returns a list of raw dictionaries."""
        ...


# ---- The Primary API ----

def read_data_from_file(
        reader: DataReadProcessProtocol,
        file_path: str,
        **kwargs: Any
) -> List[DataInput]:
    """Orchestrates data loading, schema validation, and error reporting.

    This is the recommended way to load data into the evaluator. It wraps
    the reader's output in Pydantic validation to ensure the rest of the
    pipeline receives clean, typed data.

    Args:
        reader: An implementation of DataReaderProtocol.
        file_path: Path to the source file.
        **kwargs: Configuration passed directly to the reader's read method.

    Returns:
        List[DataInput]: A list of validated Pydantic models.

    Raises:
        ValidationError: If the input data violates the DataInput schema.
        TypeError: If the provided reader does not conform to the protocol.
    """
    if not isinstance(reader, DataReadProcessProtocol):
        raise TypeError(f"Object {type(reader).__name__} does not implement DataReaderProtocol")

    logger.info(f"Loading data from {file_path} using {type(reader).__name__}")

    raw_data = reader.read(file_path, **kwargs)

    try:
        # TypeAdapter allows us to validate the entire list at once without needing to define a wrapper model
        adapter = TypeAdapter(List[DataInput])
        validated_data = adapter.validate_python(raw_data)
        logger.success(f"Successfully loaded {len(validated_data)} records from {file_path}")
        return validated_data

    except ValidationError as e:
        logger.error(f"Schema validation failed for {file_path}")
        # We re-raise to let the caller handle the specific failure
        raise
