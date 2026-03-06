"""Tests for the VLLMPredictor node.

Why:
    Ensures that the predictor correctly initializes the vLLM engine on the GPU,
    processes batched inputs via the local pipeline, and correctly maps the 
    native vLLM RequestOutputs (including token counts) to the standard 
    PredictionResult schema.

How:
    Run via terminal: `pytest test/test_predictors/test_vllm_predictor.py -s`
"""

import pytest
from vllm import SamplingParams

# Replace with the correct paths of your project
from NL2SQLEvaluator.predictor_nodes.predictor_output import PredictionResult
from NL2SQLEvaluator.dataset_reader_nodes.data_reader_protocol import ChatMessageHF
from NL2SQLEvaluator.predictor_nodes.vllm_predictor import VLLMPredictor
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_actual_vllm_inference():
    """
    Real integration test that downloads a tiny model, 
    loads it on the GPU using vLLM, and generates an output.
    WARNING: Requires a GPU and an internet connection the first time.
    """
    predictor = VLLMPredictor()
    
    # 1. Test model
    test_model_name = "HuggingFaceTB/SmolLM-135M-Instruct" 
    
    # 2. Dummy data
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
    
    # 4. Real inference execution
    results = predictor.infer(
        model_name=test_model_name,
        multiple_tasks_messages=messages,
        sampling_params=sampling_params,
        gpu_memory_utilization=0.3,
        max_model_len=512,
        tensor_parallel_size=1
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
        assert prediction.model == test_model_name, "The saved model name does not match the requested one"
        
        # Token count checks
        assert prediction.prompt_tokens is not None and prediction.prompt_tokens > 0
        assert prediction.completion_tokens is not None and prediction.completion_tokens > 0
        assert prediction.total_tokens == prediction.prompt_tokens + prediction.completion_tokens
        
        print(f"\n{'='*40}")
        print(f"--- Task {idx + 1} Result ---")
        print(f"Tokens -> Prompt: {prediction.prompt_tokens} | Completion: {prediction.completion_tokens} | Total: {prediction.total_tokens}")
        print(f"Finish Reason: {prediction.finish_reason}")
        
        print("\n--- 1. Raw Model Output ---")
        print(prediction.raw_prediction)
        
        print("\n--- 2. Parsed SQL Output ---")
        print(prediction.parsed_prediction if prediction.parsed_prediction else "No extracted SQL (None)")
        
        print("\n--- 3. Full PredictionResult Object ---")
        print(prediction.model_dump_json(indent=2))
        print(f"{'='*40}\n")