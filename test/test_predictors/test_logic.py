"""Tests for the Core Predictor Logic and SQL Extraction.

Why:
    Verifies that the regular expressions correctly extract SQL codeblocks 
    or keywords from raw text, and ensures the Pydantic validator for 
    PredictionResult triggers automatically. Tests the helper functions 
    without requiring GPU resources.

How:
    Run via terminal: `pytest test/test_predictors/test_logic.py`
"""

import pytest
from NL2SQLEvaluator.predictor_nodes.predictor_output import _extract_sql

class TestSQLExtraction:
    def test_extract_with_markdown_sql(self):
        text = "This is your query:\n```sql\nSELECT * FROM users;\n```\nHope it is useful."
        assert _extract_sql(text) == "SELECT * FROM users;"

    def test_extract_with_markdown_no_language(self):
        text = "Query:\n```\nSELECT id FROM table\n```"
        assert _extract_sql(text) == "SELECT id FROM table"

    def test_extract_with_uppercase_sql_markdown(self):
        text = "```SQL\nUPDATE users SET name = 'Mario';\n```"
        assert _extract_sql(text) == "UPDATE users SET name = 'Mario';"

    def test_extract_fallback_keywords(self):
        # No ticks, but starts with SQL keyword
        text = "SELECT count(*) FROM orders WHERE price > 10"
        assert _extract_sql(text) == "SELECT count(*) FROM orders WHERE price > 10"
        
        text_with_newline = "\n\nINSERT INTO logs (msg) VALUES ('test')"
        assert _extract_sql(text_with_newline) == "INSERT INTO logs (msg) VALUES ('test')"

    def test_extract_no_sql_present(self):
        text = "Just a regular text without any SQL."
        assert _extract_sql(text) is None

from NL2SQLEvaluator.predictor_nodes.predictor_output import PredictionResult

class TestPredictionResult:
    def test_auto_parse_triggers_on_init(self):
        # Pass the raw_prediction
        result = PredictionResult(
            raw_prediction="```sql\nSELECT name FROM clients;\n```",
            model="test-model"
        )
        # Verify that parsed_prediction has populated itself
        assert result.parsed_prediction == "SELECT name FROM clients;"
        assert result.model == "test-model"

    def test_auto_parse_does_not_override_explicit_value(self):
        # If I explicitly pass a parsed_prediction, the validator should not override it
        result = PredictionResult(
            raw_prediction="```sql\nSELECT 1;\n```",
            parsed_prediction="SELECT 2;"
        )
        assert result.parsed_prediction == "SELECT 2;"

    def test_auto_parse_with_unparsable_text(self):
        result = PredictionResult(raw_prediction="Hello World")
        assert result.parsed_prediction is None

from unittest.mock import MagicMock
from vllm import SamplingParams

from NL2SQLEvaluator.config import ModelArgs
from NL2SQLEvaluator.predictor_nodes.predictor_protocol import generate_predictions, create_sampling_params

class TestHelperFunctions:
    def test_create_sampling_params(self):
       
        mock_args = MagicMock(spec=ModelArgs)
        mock_args.number_of_completions = 3
        mock_args.repetition_penalty = 1.2
        mock_args.temperature = 0.5
        mock_args.top_p = 0.9
        mock_args.top_k = 50
        mock_args.max_new_tokens = 100

        params = create_sampling_params(mock_args)
        
        assert isinstance(params, SamplingParams)
        assert params.n == 3
        assert params.repetition_penalty == 1.2
        assert params.temperature == 0.5
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.max_tokens == 100

    def test_generate_predictions_calls_predictor(self):
        mock_predictor = MagicMock()
        mock_predictor.infer.return_value = [[PredictionResult(raw_prediction="SELECT 1")]]

        mock_args = MagicMock(spec=ModelArgs)
        mock_args.number_of_completions = 1     
        mock_args.repetition_penalty = 1.0      
        mock_args.temperature = 0.0             
        mock_args.top_p = 1.0                  
        mock_args.top_k = -1                    
        mock_args.max_new_tokens = 50           

        messages = [[{"role": "user", "content": "Test"}]]
        
        results = generate_predictions(
            predictor=mock_predictor,
            model_name="fake-model",
            multiple_tasks_messages=messages,
            model_args=mock_args,
            extra_kwarg="test"
        )

        assert len(results) == 1
        assert results[0][0].raw_prediction == "SELECT 1"
        
        mock_predictor.infer.assert_called_once()
        _, kwargs = mock_predictor.infer.call_args
        
        assert kwargs["model_name"] == "fake-model"
        assert kwargs["multiple_tasks_messages"] == messages
        assert kwargs["sampling_params"].n == 1
        assert kwargs["extra_kwarg"] == "test"