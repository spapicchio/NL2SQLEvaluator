from NL2SQLEvaluator.predictor_nodes.vllm_predictor import VLLMPredictor

try:
    from NL2SQLEvaluator.predictor_nodes.litellm_predictor import LiteLLMPredictor  # noqa: F401
except ImportError:
    pass  # litellm is an optional extra; skip registration if not installed