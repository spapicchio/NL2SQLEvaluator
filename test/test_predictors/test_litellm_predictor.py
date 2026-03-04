"""Tests for the LiteLLMPredictor node.

Why:
    Validates that the predictor successfully establishes an HTTP connection 
    to a local or remote OpenAI-compatible server (e.g., hosted vLLM), 
    handles the request parameters correctly, and parses the standardized 
    ModelResponse (and token usage) into the PredictionResult schema.

How:
    Run via terminal: `pytest test/test_predictors/test_litellm_predictor.py -s`

Note: 
    This test will fail if you don't have a vLLM server running locally on port 8000. 
    You can start one in a separate terminal using this example:
    `vllm serve HuggingFaceTB/SmolLM-135M-Instruct --max-model-len 512`
"""

import pytest
from vllm import SamplingParams

from NL2SQLEvaluator.predictor_nodes.predictor_output import PredictionResult
from NL2SQLEvaluator.dataset_reader_nodes.data_reader_protocol import ChatMessageHF
from NL2SQLEvaluator.predictor_nodes.litellm_predictor import LiteLLMPredictor

def test_actual_litellm_hosted_vllm_inference():
    """
    Real integration test that makes an actual HTTP request to a local vLLM server
    using LiteLLM.
    
    WARNING: This test will fail if you don't have a vLLM server running locally 
    on port 8000. You can start one in a separate terminal using:
    `vllm serve HuggingFaceTB/SmolLM-135M-Instruct --max-model-len 512`
    """
    predictor = LiteLLMPredictor()
    
    # 1. Test model: This MUST match the model name currently being served by your local vLLM instance
    test_model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
    
    # 2. Dummy data (multiple_tasks_messages)
    messages: list[list[ChatMessageHF]] = [
        [
            {"role": "system", "content": "You are a SQL expert. Output only the SQL query inside ```sql codeblocks."},
            {"role": "user", "content": "Write a SQL query to select all columns from the 'users' table."}
        ],
        [
            {"role": "system", "content": "You are a SQL expert. Output only the SQL query inside ```sql codeblocks."},
            {"role": "user", "content": "Count the total number of entries in the 'orders' table."}
        ]
    ]
    
    # 3. Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,      
        max_tokens=50,        
        n=1                
    )
    
    # 4. Real inference execution via HTTP
    # We specify 'hosted_vllm' and point it to localhost:8000
    results = predictor.infer(
        model_name=test_model_name,
        multiple_tasks_messages=messages,
        sampling_params=sampling_params,
        litellm_provider="hosted_vllm",
        vllm_server_host="localhost",
        vllm_server_port=8000
    )
    
    # 5. Assertions
    assert len(results) == 2, "Must return a batch of results for each submitted task (2 tasks)"
    
    for idx, task_results in enumerate(results):
        assert len(task_results) == 1, "We requested n=1 (one completion per task)"
        
        prediction = task_results[0]
        
        # Type checks
        assert isinstance(prediction, PredictionResult), f"The output is not a PredictionResult: {type(prediction)}"
        
        # Generated content checks
        assert isinstance(prediction.raw_prediction, str)
        assert len(prediction.raw_prediction.strip()) > 0, "The model generated an empty string"
        
        # Token count checks (LiteLLM should parse the 'usage' block from the OpenAI-compatible response)
        assert prediction.prompt_tokens is not None and prediction.prompt_tokens > 0, "Prompt tokens were not parsed"
        assert prediction.completion_tokens is not None and prediction.completion_tokens > 0, "Completion tokens were not parsed"
        assert prediction.total_tokens == prediction.prompt_tokens + prediction.completion_tokens
        
        print(f"\n{'='*40}")
        print(f"--- LiteLLM Task {idx + 1} Result ---")
        print(f"Model returned: {prediction.model}")
        print(f"Tokens -> Prompt: {prediction.prompt_tokens} | Completion: {prediction.completion_tokens} | Total: {prediction.total_tokens}")
        print(f"Finish Reason: {prediction.finish_reason}")
        
        print("\n--- 1. Raw Model Output ---")
        print(prediction.raw_prediction)
        
        print("\n--- 2. Parsed SQL Output ---")
        print(prediction.parsed_prediction if prediction.parsed_prediction else "No extracted SQL (None)")
        
        print("\n--- 3. Full PredictionResult Object ---")
        print(prediction.model_dump_json(indent=2))
        print(f"{'='*40}\n")