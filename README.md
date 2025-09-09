# NL2SQLEvaluator

# Roadmap
- [ ] Add MySQL database executor 
- [ ] Add Precision, Recall, F1 metrics for ambiguity Text2SQL datasets

👷🏼‍♂️ Work in progress 

# Configuration Guide

## 🔧 Use a YAML config (with CLI overrides)

Run your experiment with a config file:

```bash
nl2sql_eval --config path/to/config.yaml
```

### Example `config.yaml`

```yaml
# Core
output_dir: ./outputs
seed: 42

# Dataset
relative_db_base_path: data/bird_dev/dev_databases
dataset_path: simone-papicchio/bird
dataset_name: bird-dev

# Model
model_name: Qwen3-Coder-30B
model: Qwen/Qwen3-Coder-30B-A3B-Instruct
temperature: 0.7
top_p: 0.8
top_k: 20
repetition_penalty: 1.05
max_tokens: 32000

# Weights & Biases
project: text2sql-eval
entity: spapicchio-politecnico-di-torino   # or your team
group: evals
mode: online                               # or "offline" on clusters without net
tags: [eval, seg]
notes: ""
job_type: eval
```

### Override any value from the CLI

Command-line flags take precedence over the YAML:

```bash
nl2sql_eval --config config.yaml \
  --output_dir ./outputs/run-42 \
  --mode offline \
  --temperature 0.2 \
  --max_tokens 4096 \
  --tags eval --tags ablation
```

### Notes

* The config is **flat** (all keys at top level) so it works smoothly with the parser.
* Lists (e.g., `tags`) can be provided in YAML or by repeating the flag in CLI (`--tags ...` multiple times).
* Booleans accept `true/false` in YAML and `--flag true/false` in CLI.
* This package uses TRL’s `TRLParser` / HF’s `HfArgumentParser` under the hood, so the same configuration behaviors apply.


> This package relies on [TrlParser](https://huggingface.co/docs/trl/main/en/script_utils) so all the configurations available there can be used as well.