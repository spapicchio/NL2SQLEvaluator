from unittest.mock import MagicMock

import pytest

from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import execute_queries_in_model_predictions, DBExecutorProtocol
from NL2SQLEvaluator.db_executor_nodes.db_executor_input import TaskToBeExecuted
from NL2SQLEvaluator.db_executor_nodes.utils import utils_extract_sql_or_same


# Assuming the above code is in a file named sql_engine.py
# from sql_engine import TaskToBeExecuted, utils_extract_sql_or_same, execute_queries_in_model_predictions

class TestSQLExecutionModule:

    def test_task_params_broadcasting(self):
        """Test that params are correctly broadcasted to match query length."""
        queries = ["SELECT 1", "SELECT 2"]
        task = TaskToBeExecuted(db_path="test.db", queries=queries, params={"user": "admin"})

        assert len(task.params) == 2
        assert task.params[0] == {"user": "admin"}
        assert task.params[1] == {"user": "admin"}

    def test_task_params_length_mismatch(self):
        """Test that passing mismatched params raises a ValueError."""
        with pytest.raises(ValueError):
            TaskToBeExecuted(
                db_path="test.db",
                queries=["SELECT 1"],
                params=[{}, {}]  # 2 sets of params for 1 query
            )

    @pytest.mark.parametrize("input_text, expected", [
        ("Go to <answer>SELECT * FROM table</answer>", "SELECT * FROM table"),
        ("Here is the SQL: ```sql SELECT 1 ```", "SELECT 1"),
        ("```SELECT 2```", "SELECT 2"),
        ("Just a string", "Just a string"),
        ("<answer>```sql SELECT 3 ```</answer>", "SELECT 3"),
    ])
    def test_sql_extraction(self, input_text, expected):
        """Test extraction logic with various LLM response formats."""
        assert utils_extract_sql_or_same(input_text) == expected

    def test_execute_queries_grouping_logic(self):
        """Test that queries are correctly grouped by DB and then reassembled."""
        # Mock Executor
        mock_executor = MagicMock(DBExecutorProtocol)

        # We have 3 batches, but only 2 unique DBs
        db_files = ["db1.sqlite", "db2.sqlite", "db1.sqlite"]
        queries = [
            ["SELECT a"],  # Batch 0 (db1)
            ["SELECT b", "SELECT c"],  # Batch 1 (db2)
            ["SELECT d"]  # Batch 2 (db1)
        ]

        # The executor will see 2 tasks:
        # Task 0 (db1): ["SELECT a", "SELECT d"]
        # Task 1 (db2): ["SELECT b", "SELECT c"]
        mock_executor.execute_queries.return_value = [
            ["result_a", "result_d"],  # Results for task 0
            ["result_b", "result_c"]  # Results for task 1
        ]

        results = execute_queries_in_model_predictions(
            mock_executor, db_files, queries
        )

        assert len(results) == 3
        assert results[0] == ["result_a"]
        assert results[1] == ["result_b", "result_c"]
        assert results[2] == ["result_d"]

        # Verify the executor was called with exactly 2 tasks
        args, kwargs = mock_executor.execute_queries.call_args
        assert len(kwargs['tasks']) == 2
