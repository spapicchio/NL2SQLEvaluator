import pytest

from NL2SQLEvaluator.node_registry import (
    register_node,
    get_node_from_registry,
    get_available_functions,
    _registry,
    _package_level_name
)


class TestNodeRegistry:
    """Suite for validating the dynamic node registry functionality.

    Why:
        Ensures that the decoupling mechanism remains reliable and that
        module path inference doesn't break during refactoring.
    """

    def setup_method(self):
        """Snapshot and clear the global registry before each test to ensure isolation."""
        self._saved_registry = {k: dict(v) for k, v in _registry.items()}
        _registry.clear()

    def teardown_method(self):
        """Restore the global registry after each test."""
        _registry.clear()
        _registry.update(self._saved_registry)

    def test_register_and_get_function(self):
        """Tests if a basic function can be registered and retrieved."""

        @register_node(package_name="utils")
        def my_func():
            return "hello"

        retrieved = get_node_from_registry("utils", "my_func")
        assert retrieved == my_func
        assert retrieved() == "hello"

    def test_register_class_preserves_type(self):
        """Tests if a class is registered correctly and can be instantiated manually."""

        @register_node(package_name="models")
        class MyModel:
            def __init__(self, value):
                self.value = value

        cls = get_node_from_registry("models", "MyModel")
        instance = cls(value=42)
        assert instance.value == 42

    def test_package_inference(self):
        """Tests if the registry correctly infers the package name from the module path."""

        def dummy_node(): pass

        # Simulate: NL2SQLEvaluator.extractors.query_gen
        dummy_node.__module__ = "NL2SQLEvaluator.extractors.query_gen"

        register_node()(dummy_node)

        assert "extractors" in _registry
        assert "dummy_node" in _registry["extractors"]

    def test_get_available_functions(self):
        """Tests listing all registered keys in a package."""

        @register_node(package_name="ops")
        def op1(): pass

        @register_node(package_name="ops")
        def op2(): pass

        available = get_available_functions("ops")
        assert set(available) == {"op1", "op2"}

    def test_missing_package_raises_error(self):
        """Verifies that accessing a non-existent package raises KeyError."""
        with pytest.raises(KeyError, match="Package 'ghost' not found"):
            get_node_from_registry("ghost", "any_node")

    def test_missing_node_raises_error(self):
        """Verifies that accessing a non-existent node in a valid package raises KeyError."""

        @register_node(package_name="ops")
        def op1(): pass

        with pytest.raises(KeyError, match="Node 'op2' not found"):
            get_node_from_registry("ops", "op2")

    @pytest.mark.parametrize("module_path, expected", [
        ("NL2SQLEvaluator.db.nodes", "db"),
        ("NL2SQLEvaluator.pipeline", "pipeline"),
        ("other_pkg.sub.mod", "sub"),
        ("standalone", "standalone"),
    ])
    def test_package_level_name_logic(self, module_path, expected):
        """Tests the internal logic for extracting sub-package names."""

        class MockNode: pass

        MockNode.__module__ = module_path

        result = _package_level_name(MockNode, base_package="NL2SQLEvaluator")
        assert result == expected
