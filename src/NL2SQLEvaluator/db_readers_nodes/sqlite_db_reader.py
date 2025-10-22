import multiprocessing as mp
import sqlite3

from func_timeout import func_timeout, FunctionTimedOut

from NL2SQLEvaluator.db_readers_nodes.db_reader_protocol import OutputTable
from NL2SQLEvaluator.node_registry import register_node


@register_node()
class SQLiteDBReader:
    def execute_queries(self, db_file: str, queries: list[str], *args, **kwargs) -> list[OutputTable]:
        timeout_s = kwargs.get("timeout", 10)
        raw_results: list[OutputTable] = []
        if len(queries) == 0:
            return raw_results

        elif len(queries) == 1:
            raw_results = [SQLiteDBReader._execute_single_query(db_file, queries[0], timeout_s)]

        else:
            num_cpus = kwargs.get("num_cpus", min(max(1, mp.cpu_count()), max(1, len(queries))))
            with mp.Pool(processes=num_cpus) as pool:
                raw_results: list[OutputTable] = pool.starmap(
                    SQLiteDBReader._execute_single_query,
                    [(query, timeout_s) for query in queries],
                )

        return raw_results

    @staticmethod
    def _execute_single_query(db_file, query: str, timeout_s: float) -> OutputTable:
        """
        Executes the SQL with a timeout and returns (job_id, kind, pred_idx, rows_or_none).
        Return None in case of any error or timeout.
        """

        def _internal():
            conn = None
            try:
                conn = sqlite3.connect(db_file)
                conn.execute("PRAGMA foreign_keys=ON;")
                conn.execute("PRAGMA query_only=ON;")
                cur = conn.cursor()
                conn.execute("BEGIN TRANSACTION;")
                cur.execute(query)
                output = cur.fetchall()
            finally:
                try:
                    if conn is not None:
                        conn.rollback()
                        conn.close()
                finally:
                    pass
            return output

        try:
            rows = func_timeout(timeout_s, _internal)
        except (FunctionTimedOut, Exception):
            rows = None
        return rows
