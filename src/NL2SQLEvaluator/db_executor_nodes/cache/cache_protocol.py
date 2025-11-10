import hashlib
from typing import Self, Any, Iterator, Protocol

import sqlglot
from pydantic import BaseModel, model_validator


class NotFoundInCacheError(Exception):
    """Raised when an item is not found in the cache."""
    pass


class OutputTable(BaseModel):
    rows: list[tuple | list]
    executed_time: float | None = None

    @model_validator(mode='after')
    def forbid_inner_lists(self) -> Self:
        for i, row in enumerate(self.rows):
            for j, val in enumerate(row):
                if isinstance(val, list | tuple | set | dict):
                    raise TypeError(
                        f"rows[{i}][{j}] is a list, which is forbidden"
                    )
        return self

    def __len__(self):
        return len(self.rows)

    def __call__(self, index: int | slice | None = None) -> Any:
        """
        Call to access rows:
        - no argument -> return the full rows list
        - int -> return the row at that index (supports negative indices)
        - slice -> return a sublist of rows
        """
        if index is None:
            return self.rows
        if isinstance(index, (int, slice)):
            return self.rows[index]
        raise TypeError("index must be an int, slice, or None")

    def __getitem__(self, item):
        return self.rows[item]

    def __iter__(self) -> Iterator[Any]:
        return iter(self.rows)

    def __contains__(self, item: Any) -> bool:
        return item in self.rows

    def compress(self) -> bytes:
        """Compress the result to save space in cache."""
        # Implement compression logic if needed
        import pickle
        return pickle.dumps(self)

    def decompress(self) -> Self:
        import pickle
        return pickle.loads(self.rows[0][0]) if self.rows and self.rows[0] else self

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OutputTable):
            return self.rows == other.rows
        return NotImplemented



class DataToFetch(BaseModel):
    db_id: str
    query: str
    dialect: str = "sqlite"
    hash_key: Any = None

    @model_validator(mode='after')
    def create_hash_key(self) -> Self:
        try:
            self.query = sqlglot.transpile(self.query, self.dialect, identity=True)[0]
        except Exception:
            ...  # keep original query if transpile fails
        value = f"{self.db_id}|{self.query}"
        self.hash_key = self.hash_key or hashlib.sha256(value.encode("utf-8")).hexdigest()
        return self


class DataToCache(DataToFetch):
    result: OutputTable


class SQLCacheProtocol(Protocol):
    @staticmethod
    def set_in_cache(db_file: str, data_to_cache: list[DataToCache]) -> None:
        """Save a file with the given name and parameters."""
        ...

    @staticmethod
    def get_from_cache(db_file: str, data_to_fetch: list[DataToFetch]) -> list[DataToCache | NotFoundInCacheError]:
        """Retrieve a file with the given name and parameters."""
        ...
