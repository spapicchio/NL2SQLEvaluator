"""
SPARQL endpoint implementation of the database execution protocol.

Uses ThreadPoolExecutor for parallelism (HTTP requests are I/O-bound)
and enforces read-only by rejecting SPARQL Update keywords before execution.
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Any

from func_timeout import func_timeout, FunctionTimedOut
from SPARQLWrapper import SPARQLWrapper, JSON

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import (
    SQLCacheProtocol, DataToFetch, NotFoundInCacheError
)
from NL2SQLEvaluator.db_executor_nodes.db_executor_input import TaskToBeExecuted
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SparqlExecutorOutput, ExecutorError
from NL2SQLEvaluator.node_registry import register_node

# SPARQL Update keywords that indicate write operations.
# Only match keywords that appear OUTSIDE of URIs (<...>) and string literals ("..." / '...').
_URI_OR_LITERAL = r"""<[^>]*>|"[^"]*"|'[^']*'"""
_WRITE_KEYWORDS = r'\b(INSERT|DELETE|DROP|CLEAR|LOAD|CREATE|ADD|MOVE|COPY)\b'
_SPARQL_WRITE_PATTERN = re.compile(
    rf'{_URI_OR_LITERAL}|({_WRITE_KEYWORDS})',
    re.IGNORECASE
)


def _check_read_only(query: str) -> None:
    """Reject queries containing SPARQL Update keywords outside URIs and literals."""
    for m in _SPARQL_WRITE_PATTERN.finditer(query):
        if m.group(2):  # keyword matched outside a URI/literal
            raise ExecutorError("Write operations are not allowed. Query contains SPARQL Update keywords.")


def _execute_single_sparql(
        job_id: int,
        idx: int,
        endpoint_url: str,
        query: str,
        timeout: float,
) -> tuple[int, int, SparqlExecutorOutput | ExecutorError]:
    """Isolated worker function to execute a single SPARQL query."""
    try:
        _check_read_only(query)

        def _run_sparql() -> SparqlExecutorOutput:
            sparql = SPARQLWrapper(endpoint_url)
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            sparql.setTimeout(int(timeout))

            start = time.perf_counter()
            response = sparql.query().convert()
            elapsed = time.perf_counter() - start

            # Parse JSON results: {"results": {"bindings": [{"var": {"type": ..., "value": ...}, ...}]}}
            bindings = response.get("results", {}).get("bindings", [])
            rows = []
            for binding in bindings:
                sorted_keys = sorted(binding.keys())
                row = tuple(binding[k].get("value", str(binding[k])) for k in sorted_keys)
                rows.append(row)

            return SparqlExecutorOutput(rows=rows, execution_time=elapsed)

        result = func_timeout(timeout, _run_sparql)
        return job_id, idx, result

    except FunctionTimedOut:
        return job_id, idx, ExecutorError(f"Timeout after {timeout}s")
    except ExecutorError as e:
        return job_id, idx, e
    except Exception as e:
        return job_id, idx, ExecutorError(str(e))


@register_node(package_name="db_executor_nodes")
class SparqlEndpointExecutor:
    """Parallelized SPARQL endpoint query executor with integrated caching and safety limits."""

    def execute_queries(
            self,
            tasks: List[TaskToBeExecuted],
            cache_db: Optional[SQLCacheProtocol[SparqlExecutorOutput]] = None,
            cache_db_file: Optional[str] = None,
            **kwargs
    ) -> list[list[SparqlExecutorOutput | ExecutorError]]:
        """Executes SPARQL tasks. Checks cache first, then runs misses in parallel."""
        results: list[list[SparqlExecutorOutput | ExecutorError]] = [
            [ExecutorError("Pending execution")] * len(t.queries) for t in tasks
        ]
        pending_work = []

        # Step 1: Cache Check (Serial)
        for j_id, task in enumerate(tasks):
            for q_idx, query in enumerate(task.queries):
                if cache_db and cache_db_file:
                    cached = self._check_cache(cache_db, cache_db_file, task.db_path, query)
                    if cached is not None:
                        results[j_id][q_idx] = cached
                        continue

                timeout = task.timeout[q_idx] if isinstance(task.timeout, list) else task.timeout
                pending_work.append((j_id, q_idx, task.db_path, query, timeout))

        if len(pending_work) == 0:
            return results

        # Step 2: Parallel Execution via ThreadPoolExecutor
        max_workers = kwargs.get("max_workers", min(len(pending_work), 8))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_execute_single_sparql, *work)
                for work in pending_work
            ]
            flat_results = [f.result() for f in futures]

        # Step 3: Reassemble
        for j_id, q_idx, outcome in flat_results:
            results[j_id][q_idx] = outcome

        return results

    def _check_cache(
            self,
            cache_db: SQLCacheProtocol[SparqlExecutorOutput],
            cache_path: str,
            endpoint_url: str,
            query: str
    ) -> SparqlExecutorOutput | None:
        """Internal helper to safely probe the cache."""
        fetch_req = DataToFetch(db_path=endpoint_url, query=query, dialect="sparql")
        cache_res = cache_db.get_from_cache(cache_path, [fetch_req])[0]
        if not isinstance(cache_res, NotFoundInCacheError):
            return cache_res.result
        return None
