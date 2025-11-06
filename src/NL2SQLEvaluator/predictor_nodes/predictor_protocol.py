from typing import Protocol

from vllm import SamplingParams

from NL2SQLEvaluator.config import ModelArgs
from NL2SQLEvaluator.dataset_reader_nodes.data_reader_protocol import ChatMessageHF


class PredictorProtocol(Protocol):
    def infer(self,
              model_name: str,
              multiple_tasks_messages: list[ChatMessageHF],
              sampling_params: SamplingParams,
              *args,
              **kwargs) -> list[list[str]]:
        ...


def generate_predictions(predictor: PredictorProtocol,
                         model_name: str,
                         multiple_tasks_messages: list[ChatMessageHF],
                         model_args: ModelArgs,
                         *args,
                         **kwargs) -> list[list[str]]:
    sampling_params = create_sampling_params(model_args)
    return predictor.infer(
        model_name=model_name,
        multiple_tasks_messages=multiple_tasks_messages,
        sampling_params=sampling_params,
        *args,
        **kwargs
    )


def create_sampling_params(model_args: ModelArgs) -> SamplingParams:
    return SamplingParams(
        n=model_args.number_of_completions,
        repetition_penalty=model_args.repetition_penalty,
        temperature=model_args.temperature,
        top_p=model_args.top_p,
        top_k=model_args.top_k,
        max_tokens=model_args.max_new_tokens,
    )
