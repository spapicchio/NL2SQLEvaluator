"""Centralized registry for dynamic component discovery.

Why:
    Decouples component definitions from execution logic, allowing
    configuration-driven workflows and reducing circular imports.

How:
    >>> @register_node(package_name="db_nodes")
    >>> class MyNode: ...
    >>> node_class = get_node_from_registry("db_nodes", "MyNode")
"""

from collections.abc import Callable
from typing import Any, TypeVar

from NL2SQLEvaluator.logger import get_logger

T = TypeVar("T")

logger = get_logger(__name__)

# Internal registry storage: {package_name: {node_name: object}}
_registry: dict[str, dict[str, Any]] = {}


def register_node(
    package_name: str | None = None,
    fun_name: str | None = None,
) -> Callable[[T], T]:
    """Decorator to register a class or function into the global registry.

    Args:
        package_name: The logical group (e.g., 'extractors'). If None,
            it is inferred from the module path.
        fun_name: The lookup key. Defaults to the object's __name__.

    Returns:
        The original object, unaltered.
    """
    def decorator(node: T) -> T:
        name = fun_name or getattr(node, "__name__", str(node))
        module_key = package_name or _package_level_name(node, base_package='NL2SQLEvaluator')

        bucket = _registry.setdefault(module_key, {})
        if name in bucket:
            logger.warning("Node '%s' already exists in '%s'. Overwriting.", name, module_key)

        bucket[name] = node
        logger.debug(f"Registered node `{name}` under package `{module_key}`")
        return node

    return decorator


def get_node_from_registry(package_name: str, fun_name: str) -> Any:
    """Retrieves a registered node (class/function) from the registry.

    Args:
        package_name: The logical group name.
        fun_name: The name of the node to retrieve.

    Returns:
        The registered object (not instantiated).

    Raises:
        KeyError: If the package or node name is not found.
    """
    if package_name not in _registry:
        available = list(_registry.keys())
        raise KeyError(f"Package '{package_name}' not found. Available: {available}")

    bucket = _registry[package_name]
    if fun_name not in bucket:
        raise KeyError(f"Node '{fun_name}' not found in package '{package_name}'.")

    return bucket[fun_name]


def get_available_functions(package_name: str) -> list[str]:
    """Lists all node names registered under a specific package.

    Args:
        package_name: The logical group to query.

    Returns:
        List of registered names.
    """
    if package_name not in _registry:
        return []
    return list(_registry[package_name].keys())


def _package_level_name(node: Any, base_package: str = 'NL2SQLEvaluator') -> str:
    """Infers a logical package name from the object's module path."""
    mod = getattr(node, "__module__", "")
    parts = mod.split(".")

    # Try to find the sub-package immediately following the base_package
    if base_package in parts:
        idx = parts.index(base_package)
        if idx + 1 < len(parts):
            return parts[idx + 1]

    # Fallback to the immediate parent or the module itself
    return parts[-2] if len(parts) > 1 else (parts[0] if parts else "unknown")