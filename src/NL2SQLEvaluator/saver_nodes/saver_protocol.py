from typing import Protocol, Any

from pandas import DataFrame


class DataframeSaver(Protocol):
    @staticmethod
    def save(folder, df: DataFrame, configs: tuple[Any], *args, **kwargs):
        """Save a file with the given name and parameters."""
        pass
