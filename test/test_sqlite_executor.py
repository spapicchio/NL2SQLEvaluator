# test_sqlite_db_executor.py
import time

import pytest

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
    from sqlalchemy import create_engine, text
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
    A slow query that sleeps 0.2 s but has a 0.1 s timeout
    must come back as None, not raise.
    """
    executor = SqliteDBExecutor.from_uri(relative_base_path='/home/papicchi/NL-to-SQL-Evaluator/data/bird_dev/dev_databases/toxicology/toxicology.sqlite')
    executor.set_different_timeout(5)
    query = """
            WITH RECURSIVE ConnectedAtoms AS (
                SELECT atom_id2 AS current_atom
                FROM connected
                WHERE atom_id IN (SELECT atom_id FROM atom WHERE molecule_id = 'TR181')
                UNION ALL
                SELECT c.atom_id2
                FROM connected c
                INNER JOIN ConnectedAtoms ca ON c.atom_id = ca.current_atom
            )
            SELECT DISTINCT current_atom FROM ConnectedAtoms;
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
