"""Tests for the Data Ingestion Orchestrator.

Why:
    Ensures that the Pydantic validation logic correctly enforces
    the schema while allowing for flexible data extensions.

How:
    Run via terminal: `pytest test_ingestion.py`
"""

import pytest
from pydantic import ValidationError

from NL2SQLEvaluator.dataset_reader_nodes.data_reader_protocol import read_data_from_file, DataInput, DataReadProcessProtocol


class MockReader(DataReadProcessProtocol):
    """A helper class that conforms to DataReaderProtocol for testing."""

    def read(self, file_path: str, **kwargs) -> list:
        return kwargs.get("data", [])


class TestDataIngestion:
    """Suite to validate Pydantic-based data ingestion and Protocol enforcement."""

    def test_successful_ingestion(self):
        """Tests that valid data is correctly transformed into DataInput models."""
        valid_raw = [
            {
                "db_id": "test.db",
                "target_code": ["SELECT * FROM table"],
                "input_seq": [{"role": "user", "content": "How many users?"}]
            }
        ]
        reader = MockReader()
        results = read_data_from_file(reader, "dummy.json", data=valid_raw)

        assert len(results) == 1
        assert isinstance(results[0], DataInput)
        assert results[0].db_id == "test.db"
        assert results[0].input_seq[0].role == "user"

    def test_extra_fields_allowed(self):
        """Verifies that 'extra=allow' works for both DataInput and ChatTurn."""
        data_with_extras = [
            {
                "db_id": "test.db",
                "target_code": [],
                "input_seq": [{"role": "user", "content": "hi", "token_count": 5}],
                "researcher_notes": "this is a test"
            }
        ]
        reader = MockReader()
        results = read_data_from_file(reader, "dummy.json", data=data_with_extras)

        # Accessing extra fields via dot notation (Pydantic models) or __dict__
        assert results[0].researcher_notes == "this is a test"
        assert results[0].input_seq[0].token_count == 5

    def test_validation_failure_missing_fields(self):
        """Ensures ValidationError is raised when required keys are missing."""
        invalid_raw = [{"db_file": "missing_query_and_seq.db"}]
        reader = MockReader()

        with pytest.raises(ValidationError) as exc_info:
            read_data_from_file(reader, "bad_data.json", data=invalid_raw)

        # Verify the error message mentions the missing fields
        errors = exc_info.value.errors()
        assert any(err['loc'] == (0, 'target_code') for err in errors)
        assert any(err['loc'] == (0, 'input_seq') for err in errors)

    def test_invalid_protocol_implementation(self):
        """Tests that passing an object that doesn't fit the protocol raises TypeError."""

        class BadReader:
            # Missing the 'read' method
            pass

        with pytest.raises(TypeError, match="does not implement DataReaderProtocol"):
            read_data_from_file(BadReader(), "data.json")

    def test_chat_turn_role_validation(self):
        """Verifies that the Literal role in ChatTurn is strictly enforced."""
        bad_role_data = [
            {
                "db_id": "a.db",
                "target_code": [],
                "input_seq": [{"role": "super-admin", "content": "not allowed"}]
            }
        ]
        reader = MockReader()

        with pytest.raises(ValidationError):
            read_data_from_file(reader, "invalid_role.json", data=bad_role_data)
