from collections.abc import Callable
from typing import Any

from NL2SQLEvaluator.logger import get_logger

logger = get_logger(__name__)

_registry: dict[str, Any] = {}


def register_node() -> Callable[..., Any]:
    def decorator(node: Any) -> Any:
        name = node.__name__
        if name in _registry:
            logger.warning(f"Class '{name}' is already registered. Skipping registration.")
        else:
            _registry[name] = node
        return node

    return decorator


def get_node(name: str) -> Any:
    if name not in _registry:
        raise ValueError(f"Database reader class '{name}' is not registered.")
    return _registry[name]
