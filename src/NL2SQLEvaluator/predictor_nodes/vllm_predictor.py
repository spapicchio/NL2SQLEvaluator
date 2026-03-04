from typing import Any

from NL2SQLEvaluator.dataset_reader_nodes.data_reader_protocol import ChatMessageHF
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import register_node

from NL2SQLEvaluator.predictor_nodes.predictor_output import PredictionResult


logger = get_logger(__name__)


@register_node()
class VLLMPredictor:
    def infer(self,
              model_name: str,
              multiple_tasks_messages: list[ChatMessageHF],
              sampling_params: Any,
              *args,
              **kwargs) -> list[list[PredictionResult]]: #list[list[str]]:
        tp = kwargs.get('tensor_parallel_size', 1)
        dp = kwargs.get('data_parallel_size', 1)
        reasoning_effort = kwargs.get('reasoning_effort', None)
        max_model_len = kwargs.get('max_model_len', 8128)
        logger.info(f'Loading model {model_name} with tp={tp}, dp={dp}, max_model_len={max_model_len}.')
        llm = self._load_model(model_name, tp, dp, max_model_len)
        tokenizer = self._load_tokenizer(model_name)
        chat_prompts = tokenizer.apply_chat_template(multiple_tasks_messages,
                                                     add_generation_prompt=True,
                                                     tokenize=False,
                                                     reasoning_effort=reasoning_effort)
        output = llm.generate(chat_prompts, sampling_params)
        # responses = [o.outputs[0].text for o in output]
        # responses = [[o] if not isinstance(o, list) else o for o in responses]
        # return responses
        predictions_batch: list[list[PredictionResult]] = []
        for request_output in output:
            prompt_tokens = len(request_output.prompt_token_ids)
            completions: list[PredictionResult] = []
            
            for comp in request_output.outputs:
                completion_tokens = len(comp.token_ids)
                completions.append(
                    PredictionResult(
                        raw_prediction=comp.text,
                        finish_reason=comp.finish_reason,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        model=model_name
                    )
                )
            predictions_batch.append(completions)
            
        return predictions_batch

    def _load_model(self, model_name, tp, dp, max_model_len):
        # https://docs.vllm.ai/en/latest/configuration/optimization.html#performance-tuning-with-chunked-prefill
        from vllm import LLM
        llm = LLM(
            model=model_name,
            dtype="bfloat16",
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.92,
            swap_space=42,
            # swap_space=4,
            # disable_custom_all_reduce=True,
            trust_remote_code=True,
            data_parallel_size=dp,
            max_num_batched_tokens=16000,
        )

        return llm

    def _load_tokenizer(self, model_name):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return tokenizer
