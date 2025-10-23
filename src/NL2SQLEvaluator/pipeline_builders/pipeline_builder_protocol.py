from typing import Protocol, Any


class PipelineProtocol(Protocol):
    def run(self, config, *args, **kwargs) -> Any:
        ...

    def build_pipeline(self, config, *args, **kwargs) -> Any:
        ...
