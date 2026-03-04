"""Tests for the Pipeline Stage: _get_predictions.

Why:
    Ensures that the pipeline orchestrator correctly passes the task inputs 
    to the PredictorProtocol, unpacks the returned PredictionResult objects, 
    handles SQL parsing fallbacks, and updates the state of 
    each PipelineTask correctly. VllmPredictor is used here to validate the integration.

How:
    Run via terminal: `pytest test/test_predictors/test_get_pred.py -s`
"""

import pytest

from NL2SQLEvaluator.config import ModelArgs
from NL2SQLEvaluator.pipeline import PipelineTask, _get_predictions 

from NL2SQLEvaluator.predictor_nodes.vllm_predictor import VLLMPredictor


class TestPipelineGetPredictions:
    
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_get_predictions_actual_vllm(self):
        """
        Real integration test: uses VLLMPredictor to generate predictions 
        on GPU, passing through the '_get_predictions' pipeline stage.
        """
        # 1. Setup the real Predictor
        predictor = VLLMPredictor()
        
        # 2. Configure ModelArgs for the test
        mock_args = ModelArgs()
        mock_args.model_name_or_path = "HuggingFaceTB/SmolLM-135M-Instruct"
        mock_args.temperature = 0.0
        mock_args.max_new_tokens = 50
        mock_args.number_of_completions = 1
        mock_args.max_model_length = 2048
        
        # 3. Setup Tasks with realistic inputs
        tasks = [
            PipelineTask(
                db_path="db1.sqlite", 
                target_sqls=["SELECT * FROM users;"], 
                input_seq=[
                    {"role": "system", "content": "You are a SQL expert. Output only the SQL query inside ```sql codeblocks."},
                    {"role": "user", "content": "Write a SQL query to select all columns from the 'users' table."}
                ]
            ),
            PipelineTask(
                db_path="db2.sqlite", 
                target_sqls=["SELECT count(*) FROM orders;"], 
                input_seq=[
                    {"role": "system", "content": "You are a SQL expert. Output only the SQL query inside ```sql codeblocks."},
                    {"role": "user", "content": "Count the total number of entries in the 'orders' table."}
                ]
            )
        ]
        
        # 4. Execute the function 
        print("\n--- Starting vLLM inference via Pipeline... ---")
        updated_tasks = _get_predictions(tasks, predictor, mock_args)
        
        # 5. Assertions and Prints
        assert len(updated_tasks) == 2, "Must return exactly 2 tasks"
        
        for i, task in enumerate(updated_tasks):
            print(f"\nTask {i+1} - DB: {task.db_path}")
            print(f"Target SQL: {task.target_sqls}")
            print(f"Predicted SQL: {task.pred_sqls}")
            
            # Verify that pred_sqls has been correctly populated as a list of strings
            assert task.pred_sqls is not None, "Predictions must not be None"
            assert len(task.pred_sqls) == 1, "We requested 1 completion (n=1)"
            
            pred_string = task.pred_sqls[0]
            assert isinstance(pred_string, str), "The predicted output must have been converted to a string"
            assert len(pred_string.strip()) > 0, "The LLM generated an empty string"
            
            # Basic check to ensure parsing worked
            # (If the LLM wrote "```sql SELECT... ```", pred_string must not contain the backticks)
            assert "```" not in pred_string, "SQL parsing (or regex) failed to remove markdown backticks!"

   
    def test_get_predictions_skips_if_no_input_seq(self):
        mock_args = ModelArgs()
        mock_predictor = None # If the function exits early, we don't even need a predictor
        
        tasks = [
            PipelineTask(db_path="db1.sqlite", target_sqls=["SELECT 1;"], pred_sqls=["SELECT 1;"])
        ]
        
        updated_tasks = _get_predictions(tasks, mock_predictor, mock_args)
        
        assert updated_tasks == tasks
        assert updated_tasks[0].pred_sqls == ["SELECT 1;"]