import time

import requests
from litellm.types.utils import ModelResponse
from vllm import SamplingParams

from NL2SQLEvaluator.dataset_reader_nodes.data_reader_protocol import ChatMessageHF
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.node_registry import register_node

logger = get_logger(__name__)


def check_server_availability(base_url, total_timeout: float = 320, retry_interval: float = 60):
    """
    Check server availability with retries on failure. This function is only used with hosted_vllm provider.
    If the server is not up after the total timeout duration, raise a `ConnectionError`.
    """
    url = f"{base_url}/health/"
    start_time = time.time()  # Record the start time

    while True:
        try:
            response = requests.get(url)
        except requests.exceptions.RequestException as exc:
            # Check if the total timeout duration has passed
            elapsed_time = time.time() - start_time
            if elapsed_time >= total_timeout:
                raise ConnectionError(
                    f"The vLLM server can't be reached at {base_url} after {total_timeout} seconds. Make "
                    "sure the server is running by running `vllm serve`."
                ) from exc
        else:
            if response.status_code == 200:
                logger.info(f"Server is up at `{url}`!")
                return None

        # Retry logic: wait before trying again
        logger.info(f"Server is not up yet at `{url}`. Retrying in {retry_interval} seconds...")
        time.sleep(retry_interval)


@register_node()
class LiteLLMPredictor:
    def infer(self,
              model_name,
              multiple_tasks_messages: list[ChatMessageHF],
              sampling_params: SamplingParams,
              *args,
              **kwargs) -> list[list[str]]:
        import litellm
        # litellm._turn_on_debug()
        litellm.request_timeout = 6000  # increase request timeout to 6000 seconds
        litellm_provider = kwargs.get('litellm_provider', 'hosted_vllm')
        logger.info(f"Inferring with LiteLLM. Provider: {litellm_provider}, Model: {model_name}")
        api_base = None
        if litellm_provider == 'hosted_vllm':
            base_url = f"http://{kwargs.get('vllm_server_host', 'localhost')}:{kwargs.get('vllm_server_port', 8000)}"
            api_base = f"{base_url}/v1"
            logger.info(f'Using hosted_vllm with api_base: {api_base}')
            check_server_availability(base_url, total_timeout=500, retry_interval=60)

        # stream=True makes each item in the returned list a streaming iterator
        model_answer = litellm.batch_completion(
            model=f"{litellm_provider}/{model_name}",
            messages=multiple_tasks_messages,
            temperature=sampling_params.temperature,
            max_tokens=sampling_params.max_tokens if 'gpt' not in model_name.lower() else None,
            top_p=sampling_params.top_p,
            n=sampling_params.n,
            max_completion_tokens=sampling_params.max_tokens,
            presence_penalty=sampling_params.presence_penalty,
            frequency_penalty=sampling_params.frequency_penalty,
            api_base=api_base,
            rpm=kwargs.get('rpm', None),
            drop_params=True,
            reasoning_effort='high' if 'gpt' in model_name.lower() else None,
            max_workers=8,
        )

        parsed_responses = self.parse_model_output(model_answer)
        return parsed_responses

    def parse_model_output(
            self, model_answer: list[ModelResponse]
    ) -> list[list[str]]:

        parsed_response: list[list[str]] = []
        for out in model_answer:
            if not isinstance(out, ModelResponse):
                if isinstance(out, BaseException):
                    raise out
                ValueError(f'The output is not of type ModelResponse but type {type(out)}: {out}')

            choices_response = [choice['message']['content'] for choice in out['choices']]

            parsed_response.append(choices_response)

        parsed_response = [
            response
            for response in parsed_response
        ]

        return parsed_response
