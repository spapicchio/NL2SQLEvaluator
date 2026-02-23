"""Tests for SqliteCache with CypherExecutorOutput.

Verifies that Cypher results survive the full cache roundtrip:
compress → store → fetch → decompress, returning the correct output type.
"""

import pytest

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import DataToCache, DataToFetch, NotFoundInCacheError
from NL2SQLEvaluator.db_executor_nodes.cache.sqlite_cache import SqliteCache
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import CypherExecutorOutput


class TestSqliteCacheCypher:

    @pytest.fixture
    def cache_path(self, tmp_path):
        return str(tmp_path / "test_cache_cypher.db")

    @pytest.fixture
    def cache_instance(self, cache_path):
        return SqliteCache(cache_path)

    def test_set_and_get_success(self, cache_instance, cache_path):
        """Cypher result stored and retrieved with correct type."""
        table = CypherExecutorOutput(rows=[(30, "Alice"), (25, "Bob")])
        fetch_req = DataToFetch(
            db_path="localhost",
            query="MATCH (n:Person) RETURN n.age, n.name",
            dialect="cypher"
        )

        data_to_cache = DataToCache[CypherExecutorOutput](
            **fetch_req.model_dump(exclude={'hash_key'}),
            result=table,
        )

        cache_instance.set_in_cache(cache_path, [data_to_cache])
        results = cache_instance.get_from_cache(cache_path, [fetch_req])

        assert len(results) == 1
        assert isinstance(results[0], DataToCache)
        assert isinstance(results[0].result, CypherExecutorOutput)
        assert results[0].result.rows == table.rows

    def test_cache_miss(self, cache_instance, cache_path):
        """Missing Cypher query returns NotFoundInCacheError."""
        fetch_req = DataToFetch(
            db_path="localhost",
            query="MATCH (n) RETURN n",
            dialect="cypher"
        )
        results = cache_instance.get_from_cache(cache_path, [fetch_req])

        assert len(results) == 1
        assert isinstance(results[0], NotFoundInCacheError)

    def test_batch_operations(self, cache_instance, cache_path):
        """Multiple Cypher results cached and fetched; miss returns error."""
        table_a = CypherExecutorOutput(rows=[(1, "Gene")])
        table_b = CypherExecutorOutput(rows=[(2, "Protein")])

        data = [
            DataToCache(db_path="bio_db", query="MATCH (g:Gene) RETURN g", result=table_a, dialect="cypher"),
            DataToCache(db_path="bio_db", query="MATCH (p:Protein) RETURN p", result=table_b, dialect="cypher"),
        ]

        cache_instance.set_in_cache(cache_path, data)

        fetches = [
            DataToFetch(db_path="bio_db", query="MATCH (g:Gene) RETURN g", dialect="cypher"),
            DataToFetch(db_path="bio_db", query="MATCH (p:Protein) RETURN p", dialect="cypher"),
            DataToFetch(db_path="bio_db", query="MATCH (x:Unknown) RETURN x", dialect="cypher"),
        ]
        results = cache_instance.get_from_cache(cache_path, fetches)

        assert len(results) == 3
        assert results[0].result.rows == [(1, "Gene")]
        assert results[1].result.rows == [(2, "Protein")]
        assert isinstance(results[2], NotFoundInCacheError)

    def test_neo4j_dialect_alias(self, cache_instance, cache_path):
        """'neo4j' dialect routes to CypherExecutorOutput on retrieval."""
        table = CypherExecutorOutput(rows=[(42,)])
        fetch_req = DataToFetch(db_path="host", query="RETURN 42", dialect="neo4j")

        data_to_cache = DataToCache[CypherExecutorOutput](
            **fetch_req.model_dump(exclude={'hash_key'}),
            result=table,
        )

        cache_instance.set_in_cache(cache_path, [data_to_cache])
        results = cache_instance.get_from_cache(cache_path, [fetch_req])

        assert isinstance(results[0].result, CypherExecutorOutput)
        assert results[0].result.rows == [(42,)]
