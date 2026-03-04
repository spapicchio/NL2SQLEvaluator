import re
from typing import Any
from pydantic import BaseModel, model_validator

"""Module defining the PredictionResult data model and parsing logic for predictor outputs."""

def _extract_sql(text: str) -> str | None:
    fence_match = re.search(r"```(?:sql|SQL)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        sql = fence_match.group(1).strip()
        if sql:
            return sql

    kw_match = re.search(
        r"(?:^|\n)\s*(?:SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|DROP|ALTER)\b",
        text, re.IGNORECASE
    )
    if kw_match:
        return text[kw_match.start():].strip()

    return None

class PredictionResult(BaseModel):
    raw_prediction: str
    parsed_prediction: str | None = None
    finish_reason: str | None = None
    logprobs: list[Any] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    model: str | None = None

    @model_validator(mode="after")
    def auto_parse(self):
        if self.parsed_prediction is None and self.raw_prediction:
            self.parsed_prediction = _extract_sql(self.raw_prediction)
        return self


def parse_predictions(raw: list[list[dict[str, Any]]]) -> list[list[PredictionResult]]:
    return [
        [PredictionResult(**completion) for completion in completions]
        for completions in raw
    ]