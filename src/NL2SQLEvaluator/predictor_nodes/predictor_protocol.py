from typing import Protocol

from vllm import SamplingParams

from NL2SQLEvaluator.dataset_reader_nodes.data_readers import ChatMessageHF


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
                         sampling_params: SamplingParams,
                         *args,
                         **kwargs) -> list[list[str]]:
    return predictor.infer(
        model_name=model_name,
        multiple_tasks_messages=multiple_tasks_messages,
        sampling_params=sampling_params,
        *args,
        **kwargs
    )
