from typing import Any


class PipelineEvalOnly:
    def run(self, config, *args, **kwargs) -> Any:
        ...