from typing import Protocol

from vllm import SamplingParams

from NL2SQLEvaluator.dataset_reader_nodes.data_readers import ChatMessageHF


class PredictorProtocol(Protocol):
    def infer(self,
              model_name: str,
              multiple_tasks_messages: list[ChatMessageHF],
              sampling_params: SamplingParams,
              *args,
              **kwargs) -> list[list[str]] | list[str]:
        ...
