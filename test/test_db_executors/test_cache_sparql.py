"""Tests for SqliteCache with SparqlExecutorOutput.

Verifies that SPARQL results survive the full cache roundtrip:
compress → store → fetch → decompress, returning the correct output type.
"""

import pytest

from NL2SQLEvaluator.db_executor_nodes.cache.cache_protocol import DataToCache, DataToFetch, NotFoundInCacheError
from NL2SQLEvaluator.db_executor_nodes.cache.sqlite_cache import SqliteCache
from NL2SQLEvaluator.db_executor_nodes.db_executor_output import SparqlExecutorOutput


class TestSqliteCacheSparql:

    @pytest.fixture
    def cache_path(self, tmp_path):
        return str(tmp_path / "test_cache_sparql.db")

    @pytest.fixture
    def cache_instance(self, cache_path):
        return SqliteCache(cache_path)

    def test_set_and_get_success(self, cache_instance, cache_path):
        """SPARQL result stored and retrieved with correct type."""
        table = SparqlExecutorOutput(rows=[("http://ex.org/1", "Alice"), ("http://ex.org/2", "Bob")])
        fetch_req = DataToFetch(
            db_path="https://query.wikidata.org/sparql",
            query="SELECT ?s ?name WHERE { ?s foaf:name ?name }",
            dialect="sparql"
        )

        data_to_cache = DataToCache[SparqlExecutorOutput](
            **fetch_req.model_dump(exclude={'hash_key'}),
            result=table,
        )

        cache_instance.set_in_cache(cache_path, [data_to_cache])
        results = cache_instance.get_from_cache(cache_path, [fetch_req])

        assert len(results) == 1
        assert isinstance(results[0], DataToCache)
        assert isinstance(results[0].result, SparqlExecutorOutput)
        assert results[0].result.rows == table.rows

    def test_cache_miss(self, cache_instance, cache_path):
        """Missing SPARQL query returns NotFoundInCacheError."""
        fetch_req = DataToFetch(
            db_path="https://query.wikidata.org/sparql",
            query="SELECT ?s WHERE { ?s ?p ?o }",
            dialect="sparql"
        )
        results = cache_instance.get_from_cache(cache_path, [fetch_req])

        assert len(results) == 1
        assert isinstance(results[0], NotFoundInCacheError)

    def test_batch_operations(self, cache_instance, cache_path):
        """Multiple SPARQL results cached and fetched; miss returns error."""
        table_a = SparqlExecutorOutput(rows=[("100",)])
        table_b = SparqlExecutorOutput(rows=[("http://ex.org/Gene",)])

        data = [
            DataToCache(db_path="http://endpoint", query="SELECT (COUNT(?p) AS ?c) WHERE { ?p a :Protein }", result=table_a, dialect="sparql"),
            DataToCache(db_path="http://endpoint", query="SELECT ?g WHERE { ?g a :Gene }", result=table_b, dialect="sparql"),
        ]

        cache_instance.set_in_cache(cache_path, data)

        fetches = [
            DataToFetch(db_path="http://endpoint", query="SELECT (COUNT(?p) AS ?c) WHERE { ?p a :Protein }", dialect="sparql"),
            DataToFetch(db_path="http://endpoint", query="SELECT ?g WHERE { ?g a :Gene }", dialect="sparql"),
            DataToFetch(db_path="http://endpoint", query="SELECT ?x WHERE { ?x a :Unknown }", dialect="sparql"),
        ]
        results = cache_instance.get_from_cache(cache_path, fetches)

        assert len(results) == 3
        assert results[0].result.rows == [("100",)]
        assert results[1].result.rows == [("http://ex.org/Gene",)]
        assert isinstance(results[2], NotFoundInCacheError)
