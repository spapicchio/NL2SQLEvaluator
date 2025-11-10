import sqlite3

import pytest
from NL2SQLEvaluator.db_executor_nodes import SQLiteDBExecutor
from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import OutputTable
from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecuteTask


@pytest.fixture
def db_file(tmp_path):
    """
    Create a temporary sqlite database file under tmp_path, populate it with sample rows,
    and yield the file path as a string.
    """
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    cur.executemany(
        "INSERT INTO users (name, age) VALUES (?, ?)",
        [
            ("Alice", 30),
            ("Bob", 25),
            ("Carol", 40),
        ],
    )
    conn.commit()
    conn.close()
    yield str(db_file)
    # tmp_path cleanup is handled by pytest; no manual delete required


@pytest.fixture
def another_db_file(tmp_path) -> str:
    """
    A second DB for multi-DB scenarios.
    """
    db = tmp_path / "other.db"
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, label TEXT, price REAL)")
    cur.executemany(
        "INSERT INTO items (label, price) VALUES (?, ?)",
        [("pen", 1.2), ("notebook", 2.5), ("eraser", 0.8)],
    )
    conn.commit()
    conn.close()
    return str(db)


@pytest.fixture
def executor(db_file) -> SQLiteDBExecutor:
    return SQLiteDBExecutor()


class TestSqliteDbExecutor:
    def test_execute_query_simple(self, executor: SQLiteDBExecutor, db_file):
        task = ExecuteTask(
            db_files=[db_file],
            queries=["SELECT name FROM users WHERE age > 30"],
        )
        result = executor.execute_queries(
            tasks=[task]
        )
        assert result == [[OutputTable(rows=[("Carol",)])]]

    def test_execute_with_params(self, executor: SQLiteDBExecutor, db_file):
        task = ExecuteTask(
            db_files=[db_file],
            queries=["SELECT name FROM users WHERE age >= :min_age ORDER BY age"],
            params=[{"min_age": 30}]
        )
        result = executor.execute_queries(
            tasks=[task]
        )
        assert result == [[OutputTable(rows=[("Alice",), ("Carol",)])]]

    def test_execute_multiple_jobs_and_assembly(self, executor: SQLiteDBExecutor, db_file: str, another_db_file: str):
        # Two jobs, different DBs, multiple queries each
        task_1 = ExecuteTask(
            db_files=[db_file, db_file],
            queries=["SELECT COUNT(*) FROM users",
                     "SELECT name FROM users ORDER BY id LIMIT 1", ],
        )
        task_2 = ExecuteTask(
            db_files=[another_db_file, another_db_file],
            queries=["SELECT COUNT(*) FROM items",
                     "SELECT label FROM items ORDER BY price DESC LIMIT 1", ],
        )
        res = executor.execute_queries(
            tasks=[task_1, task_2]
        )
        # Ensure rectangular shape [jobs][idx]
        assert len(res) == 2 and len(res[0]) == 2 and len(res[1]) == 2
        assert res[0][0] == OutputTable(rows=[(3,)])  # users count
        assert res[0][1] == OutputTable(rows=[("Alice",)])  # first user by id
        assert res[1][0] == OutputTable(rows=[(3,)])  # items count
        assert res[1][1] == OutputTable(rows=[("notebook",)])  # highest price item (2.5)

    def test_empty_queries_returns_empty_list(self, executor: SQLiteDBExecutor):
        task = ExecuteTask(
            db_files=[],
            queries=[],
        )
        res = executor.execute_queries(
            tasks=[task]
        )
        assert res == [[]]

    def test_params_length_mismatch_raises(self, executor: SQLiteDBExecutor, db_file: str):
        with pytest.raises(ValueError):
            executor.execute_queries(
                tasks=[ExecuteTask(
                    db_files=[db_file],
                    queries=["SELECT 1", "SELECT 2"],
                    params=[{"x": 1}])
                ],
                timeout=5,
            )

    def test_invalid_sql_returns_executor_error(self, executor: SQLiteDBExecutor, db_file: str):
        from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecutorError
        res = executor.execute_queries(
            tasks=[ExecuteTask(
                db_files=[db_file],
                queries=["SELEC invalid FROM no_table"])]
        )
        assert isinstance(res[0][0], ExecutorError)

    def test_write_query_disallowed_by_default_raises_error(self, executor: SQLiteDBExecutor, db_file: str):
        # Your code sets PRAGMA query_only=ON when allow_write=False (default) so writes should fail.
        res = executor.execute_queries(
            tasks=[ExecuteTask(
                db_files=[db_file],
                queries=["CREATE TABLE should_fail (x INT)"])],
            timeout=5,
            num_cpus=1,
        )
        from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecutorError
        assert isinstance(res[0][0], ExecutorError)

    def test_read_then_implicit_rollback_has_no_side_effects(self, executor: SQLiteDBExecutor, db_file: str):
        # Even if a write sneaks in a SELECT transaction, your code rollbacks after SELECT branch.
        # We simulate a harmless read and then verify DB unchanged at the end.
        pre = executor.execute_queries(
            tasks=[ExecuteTask(

                db_files=[db_file],
                queries=["SELECT COUNT(*) FROM users"])],
            timeout=5,
            num_cpus=1,
        )
        assert pre == [[OutputTable(rows=[(3,)])]]

        # Attempt an INSERT using default (should fail due to query_only=ON → ExecutorError)
        insert = executor.execute_queries(
            tasks=[ExecuteTask(
                db_files=[db_file],
                queries=["INSERT INTO users (name, age) VALUES ('Zoe', 22)"])],
            timeout=5,
            num_cpus=1,
        )
        from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecutorError
        assert isinstance(insert[0][0], ExecutorError)
        # Verify no change

        post = executor.execute_queries(
            tasks=[ExecuteTask(
                db_files=[db_file],
                queries=["SELECT COUNT(*) FROM users"])],
            timeout=5,
            num_cpus=1,
        )
        assert post == [[OutputTable(rows=[(3,)])]]

    @pytest.mark.parametrize("num_cpus", [1, 2])
    def test_order_is_preserved_with_multiple_tasks(
            self, executor: SQLiteDBExecutor, db_file: str, another_db_file: str, num_cpus: int
    ):
        """
        Your code reconstructs results by (job_id, idx). This ensures deterministic layout
        even though Pool may return out-of-order. Validate shape and indexing.
        """
        res = executor.execute_queries(
            tasks=[
                ExecuteTask(
                    db_files=[db_file, another_db_file],
                    queries=["SELECT name FROM users WHERE age = 25",
                             "SELECT COUNT(*) FROM items"]
                ),
                ExecuteTask(
                    db_files=[db_file, another_db_file],
                    queries=["SELECT COUNT(*) FROM users",
                             "SELECT label FROM items WHERE price < 1.0"]
                )
            ],
            timeout=5,
            num_cpus=num_cpus,
        )
        assert res[0][0] == OutputTable(rows=[("Bob",)])
        assert res[0][1] == OutputTable(rows=[(3,)])
        assert res[1][0] == OutputTable(rows=[(3,)])
        assert res[1][1] == OutputTable(rows=[("eraser",)])

    def test_timeout(self, executor: SQLiteDBExecutor, db_file: str):
        """
        Force func_timeout to raise FunctionTimedOut so we don't rely on slow SQL tricks.
        """
        from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecutorError
        res = executor.execute_queries(
            tasks=[ExecuteTask(
                db_files=db_file,
                queries=[
                    """ WITH RECURSIVE cnt(x) AS (SELECT 1
                                                  UNION ALL
                                                  SELECT x + 1
                                                  FROM cnt
                                                  WHERE x < 10e10)
                        SELECT x
                        FROM cnt;
                    """
                ],
                timeout=10
            )],
            num_cpus=1,
        )
        assert isinstance(res[0][0], ExecutorError)

    def test_multiple_timeout(self, executor: SQLiteDBExecutor, db_file: str):
        """
        Force func_timeout to raise FunctionTimedOut so we don't rely on slow SQL tricks.
        """
        from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecutorError
        res = executor.execute_queries(
            tasks=[ExecuteTask(
                db_files=db_file,
                queries=[
                            """ WITH RECURSIVE cnt(x) AS (SELECT 1
                                                          UNION ALL
                                                          SELECT x + 1
                                                          FROM cnt
                                                          WHERE x < 10e10)
                                SELECT x
                                FROM cnt;
                            """
                        ] * 5,
                timeout=[5, 2, 10, 3, 10]
            )],

        )
        assert isinstance(res[0][0], ExecutorError)
        assert isinstance(res[0][1], ExecutorError)
        assert isinstance(res[0][2], ExecutorError)
        assert isinstance(res[0][3], ExecutorError)
        assert isinstance(res[0][4], ExecutorError)

    def test_bad_sql_resilient(self, executor: SQLiteDBExecutor, db_file: str):
        from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import ExecutorError
        res = executor.execute_queries(
            tasks=[ExecuteTask(
                db_files=db_file,
                queries=[
                    "SELECT * FROM non_existent_table",
                    "SELECT name FROM users WHERE age = 25",
                    "MALFORMED SQL STATEMENT",
                ],
            timeout=5,
            )],

            num_cpus=1,
        )
        assert isinstance(res[0][0], ExecutorError)
        assert res[0][1] == OutputTable(rows=[("Bob",)])
        assert isinstance(res[0][2], ExecutorError)

    def test_get_from_cache(self, executor, db_file, tmp_path):
        from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import DataToCache
        from NL2SQLEvaluator.db_executor_nodes import SqliteCache
        from NL2SQLEvaluator.db_executor_nodes.sqlite_db_executor import _db_id_from_path

        query = "SELECT * FROM users WHERE age > 30"
        result = OutputTable(rows=[("Alice", 35), ("Bob", 40)])

        # first save in cache
        cache_db_file = tmp_path / "cache_test.db"
        cache_db = SqliteCache()

        cache_db.set_in_cache(
            str(cache_db_file),
            data_to_cache=[DataToCache(db_id=_db_id_from_path(db_file), query=query, result=result)]
        )
        # since the actual result is different from cached, we should get the cached one
        task = ExecuteTask(
            db_files=[db_file],
            queries=[query],
        )
        res = executor.execute_queries(
            tasks=[task],
            cache_db=cache_db,
            cache_db_file=str(cache_db_file),
        )

        assert res == [[result]]
