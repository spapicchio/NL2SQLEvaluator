import pytest
from pydantic import ValidationError

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import DataToFetch, DataToCache
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SQLExecutorOutput


# Mocking a result table structure for DataToCache
class MockOutputTable:
    pass


@pytest.mark.parametrize("query_a, query_b, dialect", [
    ("SELECT * FROM users", "  select * FROM  users  ", "sqlite"),
    ("MATCH (n) RETURN n", "match (n) return n", "cypher"),
])
def test_normalization_consistency(query_a: str, query_b: str, dialect: str):
    """
    Test that queries which should be semantically identical produce the same hash.

    How: Instantiate two DataToFetch objects with varied formatting and compare hash_key.
    """
    fetch_a = DataToFetch(db_path="db_1", query=query_a, dialect=dialect)
    fetch_b = DataToFetch(db_path="db_1", query=query_b, dialect=dialect)

    assert fetch_a.query == fetch_b.query, "Queries were not normalized to the same string"
    assert fetch_a.hash_key == fetch_b.hash_key, "Hashes do not match for equivalent queries"


def test_different_db_ids_different_hashes():
    """
    Test that the same query on different databases produces unique hashes.
    """
    query = "SELECT 1"
    fetch_a = DataToFetch(db_path="db_alpha", query=query)
    fetch_b = DataToFetch(db_path="db_beta", query=query)

    assert fetch_a.hash_key != fetch_b.hash_key


def test_hash_key_is_read_only():
    """
    Ensures that hash_key cannot be manually overwritten as a field.
    """
    fetch = DataToFetch(db_path="test", query="SELECT 1")

    # Attempting to set hash_key should raise ValidationError since it is frozen
    with pytest.raises(ValidationError):
        fetch.hash_key = "manual_hash"  # pyrefly: ignore


def test_invalid_input_types():
    """
    Tests Pydantic validation for incorrect types.
    """
    with pytest.raises(ValidationError):
        # db_id should be a string
        DataToFetch(db_path=123, query="SELECT 1")  # pyrefly: ignore


def test_generic_for_data_to_cache():
    cache_item = DataToCache[SQLExecutorOutput](db_path="test", query="SELECT 1", result=SQLExecutorOutput(rows=[]))
    assert isinstance(cache_item.result, SQLExecutorOutput)
