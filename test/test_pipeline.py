import sqlite3

import pytest

from NL2SQLEvaluator.config import PipelineArgs
from NL2SQLEvaluator.pipeline import PipelineInput, run_pipeline


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


class TestPipeline:

    def test_no_predictions(self, db_file):
        pipe_input = PipelineInput(
            db_files=[db_file],
            target_sql=[["SELECT * FROM users;"]],
            predictions=[["SELECT * FROM users;"]],
            input_seq=None,
        )

        result = run_pipeline(data_input=pipe_input, pipeline_args=PipelineArgs())
        assert result.scores[0] == 1.0
