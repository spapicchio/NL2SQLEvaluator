from collections.abc import Callable

from NL2SQLEvaluator.db_readers_nodes import DbReaderProtocol
from NL2SQLEvaluator.logger import get_logger

logger = get_logger(__name__)

_registry: dict[str, type[DbReaderProtocol]] = {}


def register_db_reader() -> Callable[[type[DbReaderProtocol]], type[DbReaderProtocol]]:
    def decorator(db_reader_cls: type[DbReaderProtocol]) -> type[DbReaderProtocol]:
        name = db_reader_cls.__name__
        if name in _registry:
            logger.warning(f"Class '{name}' is already registered. Skipping registration.")
        else:
            _registry[name] = db_reader_cls
        return db_reader_cls

    return decorator


def get_db_reader_class(name: str) -> type[DbReaderProtocol]:
    if name not in _registry:
        raise ValueError(f"Database reader class '{name}' is not registered.")
    return _registry[name]
