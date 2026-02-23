"""Tests for the Cypher dataset reader."""

import json
import pytest
import pandas as pd

from NL2SQLEvaluator.dataset_reader_nodes.cypher_reader import ReadCypherData
from NL2SQLEvaluator.node_registry import get_node_from_registry


# ── Fixtures ──────────────────────────────────────────────────────────────

SAMPLE_RECORDS = [
    {
        "db_id": "biology",
        "question": "How many genes are there?",
        "Cypher": "MATCH (g:Gene) RETURN count(g)",
    },
    {
        "db_id": "nba",
        "question": "Who scored the most points?",
        "Cypher": "MATCH (p:Player) RETURN p.name ORDER BY p.points DESC LIMIT 1",
    },
]

NEO4J_INFO = {
    "sampled": {
        "biology": {"host": "localhost", "port": 15061, "username": "neo4j", "password": "cypherbench"},
        "nba": {"host": "localhost", "port": 15062, "username": "neo4j", "password": "secret"},
    }
}


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


@pytest.fixture
def neo4j_info_file(tmp_path):
    path = tmp_path / "neo4j_info.json"
    path.write_text(json.dumps(NEO4J_INFO))
    return str(path)


# ── Tests ─────────────────────────────────────────────────────────────────

class TestReadCypherData:

    def test_json_loading_and_field_mapping(self, dataset_json, neo4j_info_file):
        reader = ReadCypherData()
        records = reader.read(dataset_json, neo4j_info_path=neo4j_info_file)

        assert len(records) == 2

        rec = records[0]
        assert rec["db_id"] == "bolt://neo4j:cypherbench@localhost:15061"
        assert rec["target_code"] == ["MATCH (g:Gene) RETURN count(g)"]
        assert rec["input_seq"] == [{"role": "user", "content": "How many genes are there?"}]

    def test_csv_loading(self, dataset_csv, neo4j_info_file):
        reader = ReadCypherData()
        records = reader.read(dataset_csv, neo4j_info_path=neo4j_info_file)
        assert len(records) == 2
        assert records[1]["db_id"] == "bolt://neo4j:secret@localhost:15062"

    def test_bolt_uri_construction(self, dataset_json, neo4j_info_file):
        reader = ReadCypherData()
        records = reader.read(dataset_json, neo4j_info_path=neo4j_info_file)

        assert records[0]["db_id"] == "bolt://neo4j:cypherbench@localhost:15061"
        assert records[1]["db_id"] == "bolt://neo4j:secret@localhost:15062"

    def test_missing_graph_raises_error(self, tmp_path, neo4j_info_file):
        bad_records = [{"db_id": "nonexistent", "question": "?", "Cypher": "RETURN 1"}]
        path = tmp_path / "bad.json"
        pd.DataFrame(bad_records).to_json(path, orient="records")

        reader = ReadCypherData()
        with pytest.raises(KeyError, match="nonexistent"):
            reader.read(str(path), neo4j_info_path=neo4j_info_file)

    def test_target_code_wrapping(self, dataset_json, neo4j_info_file):
        reader = ReadCypherData()
        records = reader.read(dataset_json, neo4j_info_path=neo4j_info_file)

        for rec in records:
            assert isinstance(rec["target_code"], list)
            assert len(rec["target_code"]) == 1
            assert isinstance(rec["target_code"][0], str)

    def test_input_seq_format(self, dataset_json, neo4j_info_file):
        reader = ReadCypherData()
        records = reader.read(dataset_json, neo4j_info_path=neo4j_info_file)

        for rec in records:
            seq = rec["input_seq"]
            assert isinstance(seq, list)
            assert len(seq) == 1
            assert seq[0]["role"] == "user"
            assert isinstance(seq[0]["content"], str)

    def test_original_fields_preserved(self, dataset_json, neo4j_info_file):
        reader = ReadCypherData()
        records = reader.read(dataset_json, neo4j_info_path=neo4j_info_file)

        assert records[0]["question"] == "How many genes are there?"
        assert records[1]["Cypher"] == "MATCH (p:Player) RETURN p.name ORDER BY p.points DESC LIMIT 1"

    def test_custom_subset(self, tmp_path):
        neo4j_info = {
            "full": {
                "biology": {"host": "remote", "port": 7687, "username": "admin", "password": "pass"},
            }
        }
        info_path = tmp_path / "neo4j_info.json"
        info_path.write_text(json.dumps(neo4j_info))

        single_record = [SAMPLE_RECORDS[0]]
        data_path = tmp_path / "test.json"
        pd.DataFrame(single_record).to_json(data_path, orient="records")

        reader = ReadCypherData()
        records = reader.read(str(data_path), neo4j_info_path=str(info_path), neo4j_info_subset="full")

        assert records[0]["db_id"] == "bolt://admin:pass@remote:7687"


class TestRegistration:

    def test_reader_registered(self):
        cls = get_node_from_registry("dataset_readers", "ReadCypherData")
        assert cls is ReadCypherData
