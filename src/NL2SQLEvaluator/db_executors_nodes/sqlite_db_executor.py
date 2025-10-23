import multiprocessing as mp
import sqlite3

from NL2SQLEvaluator.db_executors_nodes import OutputTable, ExecutorError
from NL2SQLEvaluator.node_registry import register_node
from func_timeout import func_timeout, FunctionTimedOut


@register_node()
class SQLiteDBReader:
    @staticmethod
    def execute_queries(db_file: str, queries: list[str], params: list[dict] | None = None, *args, **kwargs) -> list[
        OutputTable | ExecutorError]:
        timeout_s = kwargs.get("timeout", 10)
        raw_results: list[OutputTable] = []
        if len(queries) == 0:
            return raw_results

        elif len(queries) == 1:
            raw_results = [SQLiteDBReader._execute_single_query(db_file, queries[0], timeout_s, params=params[0])]

        else:
            num_cpus = kwargs.get("num_cpus", min(max(1, mp.cpu_count()), max(1, len(queries))))
            with mp.Pool(processes=num_cpus) as pool:
                raw_results: list[OutputTable | ExecutorError] = pool.starmap(
                    SQLiteDBReader._execute_single_query,
                    [(query, timeout_s, False, param) for query, param in zip(queries, params)],
                )

        return raw_results

    @staticmethod
    def _execute_single_query(db_file,
                              query: str,
                              timeout_s: float,
                              allow_write: bool = False,
                              params: list[dict] | dict | None = None) -> OutputTable | ExecutorError:
        """
         Executes the SQL with a timeout. If allow_write is False the connection is set to query-only
         when supported and changes are always rolled back. If allow_write is True the statement may
         modify the DB and changes are committed on success. Returns result rows for SELECT-like
         statements or an integer rowcount for write statements. Returns None on error/timeout.
         """

        def _internal():
            conn = None
            try:
                conn = sqlite3.connect(db_file)
                # Always enable foreign keys
                conn.execute("PRAGMA foreign_keys=ON;")
                # If writes are not allowed, try to enable query_only (may not be supported)
                if not allow_write:
                    conn.execute("PRAGMA query_only=ON;")
                cur = conn.cursor()
                conn.execute("BEGIN TRANSACTION;")
                if params is not None and isinstance(params, list):
                    conn.executemany(query, params)
                else:
                    cur.execute(query, params)

                if allow_write:
                    # commit modifications
                    conn.commit()
                    result = cur.rowcount
                else:
                    # fetch read results and then rollback to avoid persisting any changes
                    result = cur.fetchall()
                    conn.rollback()
            except Exception as e:
                # ensure rollback on any error
                try:
                    if conn is not None:
                        conn.rollback()
                except Exception:
                    pass
                result = ExecutorError(e)

            finally:
                if conn is not None:
                    conn.close()
            return result

        try:
            rows = func_timeout(timeout_s, _internal)
        except FunctionTimedOut:
            rows = ExecutorError(f'Query Timeout with {timeout_s} seconds')
        return rows
