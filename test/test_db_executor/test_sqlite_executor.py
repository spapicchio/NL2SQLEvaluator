# test_sqlite_db_executor.py

import os
import sqlite3
import tempfile
import time
from unittest.mock import patch

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from NL2SQLEvaluator.db_executor.sqlite_executor import SqliteDBExecutor


# ---------------------------------------------------------------------------
# 1.  Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sample_db_path() -> str:
    return 'data/bird/bird_train/train_databases/airline/airline.sqlite'


@pytest.fixture(scope="session")
def executor(sample_db_path: str) -> SqliteDBExecutor:
    """Instantiate the read‑only executor *once* for the whole test run."""
    exec_ = SqliteDBExecutor.from_uri(
        relative_base_path=sample_db_path,
    )
    return exec_


# ---------------------------------------------------------------------------
# 2.  Helper to assert that a block finishes in ≤ N seconds (no deadlock!)
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0


# ---------------------------------------------------------------------------
# 3.  Tests
# ---------------------------------------------------------------------------
def test_execute_query_simple(executor: SqliteDBExecutor):
    result = executor.execute_query("SELECT COUNT(*) FROM Airlines")
    assert result == [(701352,)]


def test_execute_query_param(executor: SqliteDBExecutor):
    """Verify that bound parameters work."""
    res = executor.execute_query("SELECT COUNT(*) FROM Airlines WHERE TAIL_NUM=:num", params={"num": 'N956AN'})
    assert res == [(98,)]


def test_execute_multiple_query_order(executor: SqliteDBExecutor):
    """Results must be returned in the original order of the query list."""
    queries = [
        "SELECT COUNT(*) FROM Airlines",
        "SELECT COUNT(*) FROM Airlines WHERE TAIL_NUM=:num",
        "SELECT COUNT(*) FROM Airlines",
    ]
    params = [None, {"num": 'N956AN'}, None]
    got = executor.execute_multiple_query(queries, params)
    assert got == [
        [(701352,)],
        [(98,)],
        [(701352,)]
    ]


def test_many_concurrent_reads_no_deadlock(executor: SqliteDBExecutor):
    """
    Hammer the DB with 300 concurrent SELECTs.

    The whole batch must finish quickly, proving that
    ThreadPoolExecutor + SQLite readers do not deadlock.
    """
    queries = ["SELECT COUNT(*) FROM Airlines"] * 1000
    timer = Timer()
    executor.execute_multiple_query(queries, max_thread_num=40)
    assert timer.elapsed() < 5.0


def test_timeout_returns_none(executor: SqliteDBExecutor):
    """
    A slow query that sleeps 0.2s but has a 0.1s timeout
    must come back as None, not raise.
    """
    executor = SqliteDBExecutor.from_uri(
        relative_base_path='/home/papicchi/NL-to-SQL-Evaluator/data/bird_dev/dev_databases/toxicology/toxicology.sqlite')
    executor.set_different_timeout(5)
    query = """
            WITH RECURSIVE ConnectedAtoms AS (SELECT atom_id2 AS current_atom
                                              FROM connected
                                              WHERE atom_id IN (SELECT atom_id FROM atom WHERE molecule_id = 'TR181')
                                              UNION ALL
                                              SELECT c.atom_id2
                                              FROM connected c
                                                       INNER JOIN ConnectedAtoms ca ON c.atom_id = ca.current_atom)
            SELECT DISTINCT current_atom
            FROM ConnectedAtoms;
            """
    for query in [query] * 2:
        results = executor.execute_query(query)
        assert results is None


def test_multiple_timeout(executor: SqliteDBExecutor):
    """
    Ten slow queries in parallel, each exceeding its own timeout,
    must all return None.
    """
    executor = SqliteDBExecutor.from_uri(
        relative_base_path='data/bird_dev/dev_databases/toxicology/toxicology.sqlite'
    )
    executor.set_different_timeout(5)
    query = """
            WITH RECURSIVE ConnectedAtoms AS (SELECT atom_id2 AS current_atom \
                                              FROM connected \
                                              WHERE atom_id IN (SELECT atom_id FROM atom WHERE molecule_id = 'TR181') \
                                              UNION ALL \
                                              SELECT c.atom_id2 \
                                              FROM connected c \
                                                       INNER JOIN ConnectedAtoms ca ON c.atom_id = ca.current_atom)
            SELECT DISTINCT current_atom \
            FROM ConnectedAtoms;
            """
    queries = [query] * 5
    out = executor.execute_multiple_query(
        queries, max_thread_num=20, timeout=1
    )
    print(out)
    assert all(r is None for r in out)


def test_multi_thread_is_faster(executor: SqliteDBExecutor):
    query = """
            -- Recursively count to ten million, then aggregate
            WITH RECURSIVE cnt(n) AS (SELECT 1
                                      UNION ALL
                                      SELECT n + 1
                                      FROM cnt
                                      WHERE n < 100_000)
            SELECT SUM(n)
            FROM cnt; \
            """

    queries = [query] * 10
    timer = Timer()
    _ = executor.execute_multiple_query(queries, max_thread_num=15)
    elapsed = timer.elapsed()
    timer = Timer()
    for query in queries:
        _ = executor.execute_query(query)
    elapsed_single = timer.elapsed()
    assert elapsed < elapsed_single
    print(f"Multi-threaded execution took {elapsed:.2f}s, single-threaded took {elapsed_single:.2f}")


def test_bad_sql_resilient(executor: SqliteDBExecutor):
    """Malformed SQL should be swallowed and reported as None."""
    res = executor.execute_query("SELECT * FROM does_not_exist")
    assert res is None


class TestSqliteDBExecutor:

    @pytest.fixture
    def temp_db(self):
        """Create a temporary SQLite database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        # Create a simple test table
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
                       CREATE TABLE test_table
                       (
                           id    INTEGER PRIMARY KEY,
                           name  TEXT NOT NULL,
                           value INTEGER
                       )
                       """)
        cursor.execute("INSERT INTO test_table (name, value) VALUES (?, ?)", ("test1", 100))
        cursor.execute("INSERT INTO test_table (name, value) VALUES (?, ?)", ("test2", 200))
        conn.commit()
        conn.close()

        yield db_path

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def executor(self, temp_db):
        """Create a SqliteDBExecutor instance for testing."""
        return SqliteDBExecutor.from_uri(relative_base_path=temp_db)

    def test_from_uri_success(self, temp_db):
        """Test successful creation from URI."""
        executor = SqliteDBExecutor.from_uri(relative_base_path=temp_db)
        assert executor is not None
        assert executor.engine is not None

    def test_from_uri_file_not_found(self):
        """Test FileNotFoundError when database file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            SqliteDBExecutor.from_uri(relative_base_path="/nonexistent/path.db")

    def test_set_different_timeout(self, executor):
        """Test setting different timeout values."""
        original_timeout = executor.timeout
        new_timeout = 30
        executor.set_different_timeout(new_timeout)
        assert executor.timeout == new_timeout
        assert executor.timeout != original_timeout

    def test_connection_context_manager(self, executor):
        """Test connection context manager."""
        with executor.connection() as conn:
            assert conn is not None
            result = conn.execute(text("SELECT 1")).fetchone()
            assert result[0] == 1

    def test_execute_query_select(self, executor):
        """Test executing a SELECT query."""
        result = executor.execute_query("SELECT * FROM test_table ORDER BY id")
        assert result is not None
        assert len(result) == 2
        assert result[0] == (1, "test1", 100)
        assert result[1] == (2, "test2", 200)

    def test_execute_query_with_params(self, executor):
        """Test executing query with parameters."""
        result = executor.execute_query(
            "SELECT * FROM test_table WHERE name = :name",
            params={"name": "test1"}
        )
        assert result is not None
        assert len(result) == 1
        assert result[0] == (1, "test1", 100)

    def test_execute_query_insert(self, executor):
        """Test executing an INSERT query."""
        result = executor.execute_query(
            "INSERT INTO test_table (name, value) VALUES (:name, :value)",
            params={"name": "test3", "value": 300}
        )
        assert result == []  # INSERT returns empty list


    def test_execute_query_sql_error(self, executor):
        """Test handling SQL errors."""
        result = executor.execute_query("SELECT * FROM nonexistent_table")
        assert result is None

    def test_execute_query_sql_error_throw(self, executor):
        """Test throwing SQL errors when requested."""
        with pytest.raises(SQLAlchemyError):
            executor.execute_query(
                "SELECT * FROM nonexistent_table",
                throw_if_error=True
            )

    def test_execute_multiple_query_single(self, executor):
        """Test executing multiple queries with single query."""
        queries = ["SELECT COUNT(*) FROM test_table"]
        results = executor.execute_multiple_query(queries)
        assert len(results) == 1
        assert results[0] is not None

    def test_execute_multiple_query_multiple(self, executor):
        """Test executing multiple queries."""
        queries = [
            "SELECT COUNT(*) FROM test_table",
            "SELECT MAX(value) FROM test_table",
            "SELECT MIN(value) FROM test_table"
        ]
        results = executor.execute_multiple_query(queries, max_thread_num=2)
        assert len(results) == 3
        assert all(result is not None for result in results)

    def test_execute_multiple_query_with_params(self, executor):
        """Test executing multiple queries with parameters."""
        queries = [
            "SELECT * FROM test_table WHERE value > :min_val",
            "SELECT * FROM test_table WHERE name = :name"
        ]
        params = [{"min_val": 150}, {"name": "test1"}]
        results = executor.execute_multiple_query(queries, params=params)
        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is not None

    @patch('NL2SQLEvaluator.db_executor.sqlite_executor.event.listens_for')
    def test_install_pragmas_listener(self, mock_listens_for, temp_db):
        """Test PRAGMA installation listener."""
        executor = SqliteDBExecutor.from_uri(relative_base_path=temp_db)
        mock_listens_for.assert_called()
