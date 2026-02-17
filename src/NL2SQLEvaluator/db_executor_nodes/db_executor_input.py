from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutorError(Exception):
    """Raised when a database execution fails or times out."""
    pass


class TaskToBeExecuted(BaseModel):
    """Represents a set of queries to be executed with strictly list-based settings.

    Regardless of whether a single value or a list is provided during initialization,
    the 'params' and 'timeout' fields will always be lists of length equal to
    the 'queries' field after the object is created.
    """
    model_config = ConfigDict(extra='allow')

    db_path: str
    queries: list[str]
    db_id: str | None = None
    # Strictly hinted as lists for the rest of the application
    params: list[dict] | dict = Field(default_factory=dict)
    timeout: list[float | int] | float | int = 500

    @model_validator(mode='after')
    def broadcast_inputs(self) -> Self:
        """Ensures params and timeout match the number of queries.

        If params or timeouts are provided as single values, they are replicated
        to create a list of length equal to the number of queries.

        Returns:
            Self: The validated and potentially modified model instance.

        Raises:
            ValueError: If list lengths do not match the number of queries.
        """
        num_queries = len(self.queries)

        # Handle Params Broadcasting
        if isinstance(self.params, (dict, tuple)):
            self.params = [self.params for _ in range(num_queries)]

        # Handle Timeout Broadcasting
        if isinstance(self.timeout, (int, float)):
            self.timeout = [float(self.timeout) for _ in range(num_queries)]

        # Final Validation
        if len(self.params) != num_queries:
            raise ValueError(f"Params length ({len(self.params)}) != queries ({num_queries})")

        if len(self.timeout) != num_queries:
            raise ValueError(f"Timeout length ({len(self.timeout)}) != queries ({num_queries})")

        return self
