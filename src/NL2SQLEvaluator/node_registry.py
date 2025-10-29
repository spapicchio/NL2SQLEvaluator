from collections.abc import Callable
from typing import Any

from NL2SQLEvaluator.logger import get_logger

logger = get_logger(__name__)

_registry: dict[str, dict[str, Any]] = {}


def register_node(
        package_name: str | None = None,
        fun_name: str | None = None,
) -> Callable[..., Any]:
    def decorator(node: Any) -> Any:
        name = fun_name or getattr(node, "__name__", repr(node))
        module_key = package_name or _package_level_name(node, base_package='NL2SQLEvaluator')

        bucket = _registry.setdefault(module_key, {})
        if name in bucket:
            logger.warning("Class '%s' is already registered in '%s'. Skipping.", name, module_key)
        else:
            bucket[name] = node
        logger.debug(f"Registered node `{name}` under package `{module_key}`")
        return node

    return decorator


def get_node_from_registry(package_name, fun_name: str) -> Any:
    if package_name not in _registry:
        raise KeyError(f"Package '{package_name}' not found in registry.  Available packages: {list(_registry.keys())}")
    bucket = _registry[package_name]
    if fun_name not in bucket:
        raise KeyError(f"Function/Class '{fun_name}' not found in package '{package_name}'.")
    return bucket[fun_name]()


def get_available_functions(package_name) -> list[str]:
    if package_name not in _registry:
        raise KeyError(f"Package '{package_name}' not found in registry. Available packages: {list(_registry.keys())}")
    return list(_registry[package_name].keys())


def _package_level_name(node: Any, *, base_package: str | None = None) -> str:
    """
    Given a class/function, return the top-level subpackage below `base_package`
    (or the immediate parent package if base_package is None).

    Examples:
    - node.__module__ = 'NL2SQLEvaluator.db_executor_nodes.db_executor_protocol'
      base_package='NL2SQLEvaluator'  -> 'db_executor_nodes'
    - node.__module__ = 'pkg.sub.mod' base_package=None -> 'sub'
    """
    mod = getattr(node, "__module__", "") or ""
    # Remove the leaf module name to get the parent package path
    # e.g., 'a.b.c' -> 'a.b'
    parent_pkg = mod.rsplit(".", 1)[0] if "." in mod else mod

    if base_package:
        prefix = base_package + "."
        if parent_pkg.startswith(prefix):
            # Keep only the part after 'base_package.'
            rest = parent_pkg[len(prefix):]  # e.g., 'db_executor_nodes'
            return rest.split(".", 1)[0] or base_package
        # If it doesn't start with base, just fall back to last segment
    # No base_package: return immediate parent segment
    return parent_pkg.split(".")[-1] if parent_pkg else mod
