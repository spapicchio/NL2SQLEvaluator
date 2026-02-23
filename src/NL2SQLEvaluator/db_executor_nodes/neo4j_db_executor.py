"""
Neo4j implementation of the database execution protocol.

Uses ThreadPoolExecutor for parallelism (Neo4j driver is thread-safe with
connection pooling) and enforces read-only execution via session.execute_read().
"""
import atexit
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Any
from urllib.parse import urlparse

import neo4j
from neo4j import GraphDatabase, Query

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import (
    SQLCacheProtocol, DataToFetch, NotFoundInCacheError
)
from NL2SQLEvaluator.db_executor_nodes.db_executor_input import TaskToBeExecuted
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import CypherExecutorOutput, ExecutorError
from NL2SQLEvaluator.node_registry import register_node

# Module-level driver cache: keyed by "host:port" for reuse across queries
_driver_cache: dict[str, neo4j.Driver] = {}
_driver_lock = threading.Lock()


def _close_all_drivers():
    """Close all cached Neo4j drivers on interpreter shutdown."""
    with _driver_lock:
        for driver in _driver_cache.values():
            try:
                driver.close()
            except Exception:
                pass
        _driver_cache.clear()


atexit.register(_close_all_drivers)


def _parse_bolt_uri(uri: str) -> dict[str, Any]:
    """Parse a Neo4j bolt URI into components.

    Supports formats like:
        bolt://user:pass@host:7687/dbname
        bolt://host:7687/dbname
        bolt://host:7687
    """
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 7687
    username = parsed.username
    password = parsed.password
    database = parsed.path.strip("/") if parsed.path and parsed.path != "/" else None

    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "database": database,
        "bolt_url": f"{parsed.scheme or 'bolt'}://{host}:{port}",
    }


def _get_or_create_driver(bolt_url: str, username: str | None, password: str | None) -> neo4j.Driver:
    """Get or create a cached Neo4j driver instance. Thread-safe."""
    with _driver_lock:
        if bolt_url not in _driver_cache:
            auth = (username, password) if username and password else None
            driver = GraphDatabase.driver(bolt_url, auth=auth)
            driver.verify_connectivity()
            _driver_cache[bolt_url] = driver
        return _driver_cache[bolt_url]


def _execute_single_cypher(
        job_id: int,
        idx: int,
        uri: str,
        query: str,
        timeout: float,
) -> tuple[int, int, CypherExecutorOutput | ExecutorError]:
    """Isolated worker function to execute a single Cypher query."""
    try:
        parsed = _parse_bolt_uri(uri)
        driver = _get_or_create_driver(parsed["bolt_url"], parsed["username"], parsed["password"])

        session_kwargs: dict[str, Any] = {}
        if parsed["database"]:
            session_kwargs["database"] = parsed["database"]

        def tx_func(tx: neo4j.ManagedTransaction) -> list[dict]:
            result = tx.run(Query(query, timeout=timeout))
            return result.data()

        start = time.perf_counter()
        with driver.session(**session_kwargs) as session:
            records = session.execute_read(tx_func)
        elapsed = time.perf_counter() - start

        # Convert list[dict] → list[tuple]: sort each dict by key, extract values
        rows = [
            tuple(record[k] for k in sorted(record.keys()))
            for record in records
        ]

        return job_id, idx, CypherExecutorOutput(rows=rows, execution_time=elapsed)

    except (neo4j.exceptions.CypherSyntaxError,
            neo4j.exceptions.CypherTypeError,
            neo4j.exceptions.DatabaseError,
            neo4j.exceptions.ClientError) as e:
        return job_id, idx, ExecutorError(str(e))
    except neo4j.exceptions.ServiceUnavailable as e:
        return job_id, idx, ExecutorError(f"Connection failed: {e}")
    except Exception as e:
        timeout_indicators = ("timeout", "timed out", "deadline")
        if any(ind in str(e).lower() for ind in timeout_indicators):
            return job_id, idx, ExecutorError(f"Timeout after {timeout}s")
        return job_id, idx, ExecutorError(str(e))


@register_node(package_name="db_executor_nodes")
class Neo4jDBExecutor:
    """Parallelized Neo4j Cypher query executor with integrated caching and safety limits."""

    def execute_queries(
            self,
            tasks: List[TaskToBeExecuted],
            cache_db: Optional[SQLCacheProtocol[CypherExecutorOutput]] = None,
            cache_db_file: Optional[str] = None,
            **kwargs
    ) -> list[list[CypherExecutorOutput | ExecutorError]]:
        """Executes Cypher tasks. Checks cache first, then runs misses in parallel."""
        results: list[list[CypherExecutorOutput | ExecutorError]] = [
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
                pool.submit(_execute_single_cypher, *work)
                for work in pending_work
            ]
            flat_results = [f.result() for f in futures]

        # Step 3: Reassemble
        for j_id, q_idx, outcome in flat_results:
            results[j_id][q_idx] = outcome

        return results

    def _check_cache(
            self,
            cache_db: SQLCacheProtocol[CypherExecutorOutput],
            cache_path: str,
            db_path: str,
            query: str
    ) -> CypherExecutorOutput | None:
        """Internal helper to safely probe the cache."""
        parsed = _parse_bolt_uri(db_path)
        db_id = parsed["database"] or parsed["host"]
        fetch_req = DataToFetch(db_path=db_id, query=query, dialect="cypher")
        cache_res = cache_db.get_from_cache(cache_path, [fetch_req])[0]
        if not isinstance(cache_res, NotFoundInCacheError):
            return cache_res.result
        return None
