import pytest

from NL2SQLEvaluator.db_executor.base_db_executor import MySQLCache


@pytest.fixture(scope="session")  # noqa: F821
def executor() -> MySQLCache:
    """Instantiate the read‑only executor *once* for the whole test run."""
    exec_ = MySQLCache.from_uri(port=3307)
    return exec_


def test_insert_in_cache(executor: MySQLCache):
    uri = "test_db_uri"
    query = "SELECT * FROM yolewjwde"
    # Insert data into cache
    executor.insert_in_cache(
        uri, query, [("row1_col1", "row1_col2"), ("row2_col1", "row2_col2")]
    )


def test_get_from_cache(executor: MySQLCache):
    uri = "test_db_uri"
    query = "SELECT * FROM yolewjwde"
    # Insert data into cache
    executor.insert_in_cache(
        uri, query, [("row1_col1", "row1_col2"), ("row2_col1", "row2_col2")]
    )
    # Retrieve data from cache
    result = executor.get_from_cache(uri, query)
    assert result == [("row1_col1", "row1_col2"), ("row2_col1", "row2_col2")]



def test_timeout():
    executor = MySQLCache.from_uri(port=3307, timeout=0.001)
    assert executor is None
