"""Tests for the SPARQL dataset reader."""

import pytest
import pandas as pd

from NL2SQLEvaluator.dataset_reader_nodes.sparql_reader import ReadSparqlData
from NL2SQLEvaluator.node_registry import get_node_from_registry


# ── Fixtures ──────────────────────────────────────────────────────────────

ENDPOINT_URL = "http://localhost:8890/sparql"

SAMPLE_RECORDS = [
    {
        "question": "How many proteins are in the database?",
        "SPARQL": "SELECT (COUNT(?p) AS ?count) WHERE { ?p a :Protein }",
    },
    {
        "question": "Which genes are associated with cancer?",
        "SPARQL": "SELECT ?gene WHERE { ?gene :associatedWith :Cancer }",
    },
]


@pytest.fixture
def dataset_json(tmp_path):
    path = tmp_path / "test.json"
    pd.DataFrame(SAMPLE_RECORDS).to_json(path, orient="records")
    return str(path)


@pytest.fixture
def dataset_csv(tmp_path):
    path = tmp_path / "test.csv"
    pd.DataFrame(SAMPLE_RECORDS).to_csv(path, index=False)
    return str(path)


# ── Tests ─────────────────────────────────────────────────────────────────

class TestReadSparqlData:

    def test_json_loading_and_field_mapping(self, dataset_json):
        reader = ReadSparqlData()
        records = reader.read(dataset_json, sparql_endpoint_url=ENDPOINT_URL)

        assert len(records) == 2

        rec = records[0]
        assert rec["db_id"] == ENDPOINT_URL
        assert rec["target_code"] == ["SELECT (COUNT(?p) AS ?count) WHERE { ?p a :Protein }"]
        assert rec["input_seq"] == [{"role": "user", "content": "How many proteins are in the database?"}]

    def test_csv_loading(self, dataset_csv):
        reader = ReadSparqlData()
        records = reader.read(dataset_csv, sparql_endpoint_url=ENDPOINT_URL)
        assert len(records) == 2
        assert records[1]["db_id"] == ENDPOINT_URL

    def test_all_records_share_same_endpoint(self, dataset_json):
        reader = ReadSparqlData()
        records = reader.read(dataset_json, sparql_endpoint_url=ENDPOINT_URL)

        for rec in records:
            assert rec["db_id"] == ENDPOINT_URL

    def test_target_code_wrapping(self, dataset_json):
        reader = ReadSparqlData()
        records = reader.read(dataset_json, sparql_endpoint_url=ENDPOINT_URL)

        for rec in records:
            assert isinstance(rec["target_code"], list)
            assert len(rec["target_code"]) == 1
            assert isinstance(rec["target_code"][0], str)

    def test_input_seq_format(self, dataset_json):
        reader = ReadSparqlData()
        records = reader.read(dataset_json, sparql_endpoint_url=ENDPOINT_URL)

        for rec in records:
            seq = rec["input_seq"]
            assert isinstance(seq, list)
            assert len(seq) == 1
            assert seq[0]["role"] == "user"
            assert isinstance(seq[0]["content"], str)

    def test_original_fields_preserved(self, dataset_json):
        reader = ReadSparqlData()
        records = reader.read(dataset_json, sparql_endpoint_url=ENDPOINT_URL)

        assert records[0]["question"] == "How many proteins are in the database?"
        assert records[1]["SPARQL"] == "SELECT ?gene WHERE { ?gene :associatedWith :Cancer }"


class TestRegistration:

    def test_reader_registered(self):
        cls = get_node_from_registry("dataset_readers", "ReadSparqlData")
        assert cls is ReadSparqlData
