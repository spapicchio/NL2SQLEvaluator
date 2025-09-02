# python
from sqlalchemy import MetaData, Table, Column, Integer, String, Text
from sqlalchemy.dialects.sqlite import dialect as SQLiteDialect

from NL2SQLEvaluator.db_executor.utils_ddl import utils_augment_ddl


def test_returns_original_ddl_when_strategy_is_none():
    ddl = "CREATE TABLE t (\n  id INTEGER\n);"
    metadata = MetaData()
    table = Table("t", metadata, Column("id", Integer))

    result = utils_augment_ddl(
        ddl=ddl,
        table=table,
        execute_fn=lambda **kwargs: [(1,)],
        dialect=SQLiteDialect(),
        strategy=None,
        num_rows=1,
    )

    assert result == ddl


def test_returns_original_ddl_when_unknown_strategy():
    ddl = "CREATE TABLE t (\n  id INTEGER\n);"
    metadata = MetaData()
    table = Table("t", metadata, Column("id", Integer))

    result = utils_augment_ddl(
        ddl=ddl,
        table=table,
        execute_fn=lambda **kwargs: [(1,)],
        dialect=SQLiteDialect(),
        strategy="unknown",
        num_rows=1,
    )

    assert result == ddl


def test_inline_injects_example_comments_for_columns_with_non_null_samples_and_preserves_commas():
    ddl = (
        "CREATE TABLE t (\n"
        "  id INTEGER,\n"
        "  name VARCHAR(100),\n"
        "  note TEXT,\n"
        "  PRIMARY KEY (id)\n"
        ");"
    )
    metadata = MetaData()
    table = Table(
        "t",
        metadata,
        Column("id", Integer),
        Column("name", String(100)),
        Column("note", Text),
    )

    rows = [
        (1, "alice", None),
        (2, "bob", None),
    ]

    result = utils_augment_ddl(
        ddl=ddl,
        table=table,
        execute_fn=lambda **kwargs: rows,
        dialect=SQLiteDialect(),
        strategy="inline",
        num_rows=2,
    )

    lines = result.splitlines()
    id_line = next(l for l in lines if l.strip().startswith("id INTEGER"))
    name_line = next(l for l in lines if l.strip().startswith("name VARCHAR(100)"))
    note_line = next(l for l in lines if l.strip().startswith("note TEXT"))

    assert "-- Example Values:" in id_line and id_line.rstrip().endswith(",")
    assert "-- Example Values:" in name_line and name_line.rstrip().endswith(",")
    assert "alice" in name_line and "bob" in name_line
    assert "-- Example Values:" not in note_line


def test_inline_returns_original_when_no_samples():
    ddl = (
        "CREATE TABLE t (\n"
        "  id INTEGER,\n"
        "  name VARCHAR(100)\n"
        ");"
    )
    metadata = MetaData()
    table = Table(
        "t",
        metadata,
        Column("id", Integer),
        Column("name", String(100)),
    )

    result = utils_augment_ddl(
        ddl=ddl,
        table=table,
        execute_fn=lambda **kwargs: [],
        dialect=SQLiteDialect(),
        strategy="inline",
        num_rows=1,
    )

    assert result == ddl


def test_same_ddl_when_execute_fn_raises_error():
    ddl = "CREATE TABLE t (\n  id INTEGER\n);"
    metadata = MetaData()
    table = Table("t", metadata, Column("id", Integer))

    def failing_execute_fn(**kwargs):
        raise RuntimeError("DB error")

    result = utils_augment_ddl(
        ddl=ddl,
        table=table,
        execute_fn=failing_execute_fn,
        dialect=SQLiteDialect(),
        strategy="inline",
        num_rows=1,
    )
    assert result == ddl


def test_append_appends_insert_statements_for_each_row_with_literal_values_and_truncation():
    ddl = (
        "CREATE TABLE t (\n"
        "  id INTEGER,\n"
        "  name VARCHAR(100),\n"
        "  note TEXT\n"
        ");"
    )
    metadata = MetaData()
    table = Table(
        "t",
        metadata,
        Column("id", Integer),
        Column("name", String(100)),
        Column("note", Text),
    )

    long_text = "x" * 150
    rows = [
        (1, "alice", "hello"),
        (2, "bob", long_text),
    ]

    result = utils_augment_ddl(
        ddl=ddl,
        table=table,
        execute_fn=lambda **kwargs: rows,
        dialect=SQLiteDialect(),
        strategy="append",
        num_rows=2,
    )

    assert result.startswith(ddl)
    assert result.count("INSERT INTO") == 2
    assert "alice" in result and "bob" in result and "hello" in result
    assert "x" * 100 in result
    assert "x" * 101 not in result
