from dataclasses import field, dataclass

import NL2SQLEvaluator.dataset_reader_nodes  # noqa: F401
import NL2SQLEvaluator.evaluator_nodes  # noqa: F401
import NL2SQLEvaluator.predictor_nodes  # noqa: F401
import NL2SQLEvaluator.saver_nodes  # noqa: F401
from NL2SQLEvaluator.node_registry import get_available_functions


@dataclass
class ScriptArgs:
    output_dir: str | None = field(
        default="./outputs",
        metadata={"help": "Directory where evaluation outputs will be saved"}
    )
    cache_db_file_path: str | None = field(
        default="./sql_cache.sqlite",
        metadata={"help": "Path to the SQLite database file for caching SQL queries"}
    )
    num_of_experiments_to_get_std: int = field(
        default=1,
        metadata={"help": "Number of experiment runs to compute standard deviation over"}
    )
    execution_timeout: int = field(
        default=500,
        metadata={"help": "Timeout in seconds for executing SQL queries against the database"}
    )


@dataclass
class DatasetArgs:
    dataset_path: str | None = field(
        default="data/omnisql/data/train_bird_processed.json",
    )
    relative_db_base_path: str = field(
        default="data/omnisql/data/bird/train/train_databases",
        metadata={"help": "Relative path to the database files directory"}
    )
    pred_col_name: str = field(
        default=None,
        metadata={"help": "Name of the column containing predictions in the dataset"}
    )
    input_seq_col_name: str = field(
        default="input_seq",
        metadata={"help": "Name of the column containing input sequences in the dataset"}
    )
    db_id_col_name: str = field(
        default="db_id",
        metadata={"help": "Name of the column containing database identifiers in the dataset"}
    )
    target_seq_col_name: str = field(
        default='SQL',
        metadata={"help": "Name of the column containing target SQL queries in the dataset"}
    )


@dataclass
class ModelArgs:
    model_name_or_path: str = field(
        default="gpt-3.5-turbo",
        metadata={"help": "Name or path of the language model to use for SQL generation"}
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Sampling temperature for the language model"}
    )
    number_of_completions: int = field(
        default=1,
        metadata={"help": "Number of completions to generate per prompt."},
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature; lower is more deterministic (≈0.1–0.7)."},
    )
    top_p: float = field(
        default=0.8,
        metadata={"help": "Nucleus sampling: keep smallest set of tokens whose cumulative prob ≥ top_p (0–1]."},
    )
    top_k: int = field(
        default=20,
        metadata={"help": "Top-k sampling: consider only the top_k most likely tokens (0 disables)."},
    )
    repetition_penalty: float = field(
        default=1.1,
        metadata={"help": "Penalty >1.0 discourages repeating the same tokens."},
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={"help": "Downweights tokens that appear frequently in the output so far (0–2, often 0 for code)."},
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Penalizes tokens already present; encourages exploring new tokens (0–2, often 0 for code)."},
    )
    max_new_tokens: int = field(
        default=8000,
        metadata={"help": "Maximum number of new tokens to generate."},
    )
    dtype: str = field(
        default="bfloat16",
        metadata={"help": "Computation dtype (e.g., float16, bfloat16, float32, auto)."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of GPUs to shard weights across (tensor parallel)."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of data-parallel replicas (usually handled by launcher)."},
    )
    vllm_server_host: str = field(
        default="localhost",
        metadata={"help": "vLLM server host (if using vLLM server)."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "vLLM server port (if using vLLM server)."},
    )
    litellm_provider: str | None = field(
        default=None,
        metadata={"help": "Optional provider for LiteLLM. If vllm online serving, use `hosted_vllm`"},
    )
    gpu_memory_utilization: float = field(
        default=0.80,
        metadata={"help": "Target fraction of GPU memory to use (vLLM/Accelerate style runtimes)."},
    )
    max_model_length: int = field(
        default=30_000,
        metadata={"help": "Max total context (prompt + generated) tokens to allow."},
    )


@dataclass
class PipelineArgs:
    dataset_reader_node: str = field(
        default="bird_dev",
        metadata={
            "help": f"Dataset reader node to use for loading the dataset. Available readers:  {get_available_functions('dataset_reader_nodes')}"}
    )

    predictor_node: str | None = field(
        default=None,
        metadata={
            "help": f"Predictor node to use for generating predictions. Available predictors: {get_available_functions('predictor_nodes')} "}
    )
    db_executor_node: str = field(
        default="SQLiteDBExecutor",
        metadata={
            "help": f"Database executor node to use for executing SQL queries. Available executors: {get_available_functions('db_executor_nodes')}"}
    )
    sql_cache_node: str = field(
        default="SqliteCache",
        metadata={
            "help": f"SQL cache node to use for caching generated SQL queries. Available cache nodes: {get_available_functions('db_executor_nodes')}"}
    )
    evaluator_node: str = field(
        default="BirdEXEvaluator",
        metadata={
            "help": f"Evaluator node to use for evaluating predictions. Available evaluators: {get_available_functions('evaluator_nodes')}"}
    )
    saver_node: str | None = field(
        default=None,
        metadata={
            "help": f"Saver node to use for saving evaluation results. Available savers: {get_available_functions('saver_nodes')}"}
    )
