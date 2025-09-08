# args.py
import pathlib
from dataclasses import dataclass, field
from time import gmtime, strftime
from typing import Literal

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

from NL2SQLEvaluator.hf_argument_parser import TrlParser
from NL2SQLEvaluator.utils import utils_init_wandb, WandbArgs

load_dotenv(override=True)


@dataclass
class TopArgs:
    output_dir: str = field(
        default="./outputs",
        metadata={"help": "Directory where evaluation outputs will be saved"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"}
    )


@dataclass
class DatasetArgs:
    relative_db_base_path: str = field(
        default="data/bird_dev/dev_databases",
        metadata={"help": "Relative path to the database files directory"}
    )
    dataset_path: str = field(
        default="simone-papicchio/bird",
        metadata={"help": "HuggingFace dataset path or local dataset path"}
    )
    dataset_name: str = field(
        default="bird-dev",
        metadata={"help": "Name of the dataset configuration to use"}
    )
    split: Literal["train", "dev", "test"] = field(
        default="dev",
        metadata={"help": "Dataset split to evaluate on (train, dev, or test)"}
    )


@dataclass
class ModelArgs:
    model_name: str = field(
        default="Qwen3-Coder-30B",
        metadata={"help": "Human-readable name for the model"}
    )
    model: str = field(
        default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        metadata={"help": "Model identifier for HuggingFace or local model path"}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature for text generation (0.0-2.0)"}
    )
    top_p: float = field(
        default=0.8,
        metadata={"help": "Top-p (nucleus) sampling parameter (0.0-1.0)"}
    )
    top_k: int = field(
        default=20,
        metadata={"help": "Top-k sampling parameter (number of tokens to consider)"}
    )
    repetition_penalty: float = field(
        default=1.05,
        metadata={"help": "Penalty for token repetition (1.0 = no penalty)"}
    )
    max_tokens: int = field(
        default=32000,
        metadata={"help": "Maximum number of tokens to generate"}
    )


def run_evaluation(script_args, dataset_args, model_args):
    """
    Returns:
      summary metrics (scalars) and a per-sample dataframe.
    Replace with your actual evaluation routine.
    """
    seed = script_args.seed
    rng = np.random.default_rng(seed)
    n = 50

    sample_ids = [f"case_{i:04d}" for i in range(n)]
    dice = rng.uniform(0.70, 0.92, n)
    iou = dice / (2 - dice)
    density = rng.choice(["A", "B", "C", "D"], n)
    agegrp = rng.choice(["<40", "40-60", ">60"], n)

    df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "subgroup_age": agegrp,
            "subgroup_density": density,
            "dice": dice.astype(float),
            "iou": iou.astype(float),
        }
    )

    grp = df.groupby(["subgroup_age", "subgroup_density"]).agg(dice_mean=("dice", "mean"))
    dice_worst_group = float(grp.dice_mean.min())
    dice_gap_max = float(grp.dice_mean.max() - grp.dice_mean.min())

    summary = {
        "dice_mean": float(df.dice.mean()),
        "dice_ci95": float(1.96 * df.dice.std(ddof=1) / np.sqrt(n)),
        "iou_mean": float(df.iou.mean()),
        "dice_worst_group": dice_worst_group,
        "dice_gap_max": dice_gap_max,
    }
    return summary, df


def main(script_args, dataset_args, model_args, wandb_args):
    # Init W&B (respect wandb.mode: online|offline|disabled)
    wandb_run = utils_init_wandb(wandb_args, run_name=f"eval__{model_args.model_name}__{dataset_args.dataset_name}")

    # --- run your evaluation ---
    summary, df_samples = run_evaluation(script_args, dataset_args, model_args)

    run_dir = pathlib.Path(script_args.output_dir)
    today = str(strftime("%Y-%m-%d", gmtime()))
    hours_minutes = str(strftime("%H-%M", gmtime()))
    run_dir = run_dir / today / hours_minutes / f"{model_args.model_name}_{dataset_args.dataset_name}_s{script_args.seed}"
    # if run_dir does not exist create it
    run_dir.mkdir(parents=True, exist_ok=True)

    if wandb_run is not None:
        table = wandb.Table(columns=list(df_samples.columns))
        for row in df_samples.itertuples(index=False):
            table.add_data(*row)
        wandb.log({"table/dataset": table, **summary})
        # Save per-sample as Parquet and log as an Artifact
        out_parquet = run_dir / "dataset.parquet"
        df_samples.to_parquet(out_parquet, index=False)
        art = wandb.Artifact(
            name=f"dataset_eval__{dataset_args.dataset_name}__{model_args.model_name}__{wandb_run.id}",
            type="dataset",
            metadata={
                "wandb_id": wandb_run.id,
                "model": model_args.model_name,
                "data": dataset_args.dataset_name,
            },
        )
        art.add_file(str(out_parquet))
        wandb_run.log_artifact(art)

        # Store the exact resolved config that produced this run
        cfg_path = save_used_cfg(run_dir, script_args, dataset_args, model_args, wandb_args)
        cfg_art = wandb.Artifact(name=f"config_wandb_id_{wandb_run.id}", type="config")
        cfg_art.add_file(str(cfg_path))
        wandb_run.log_artifact(cfg_art)
        wandb_run.finish()
    else:
        # Even with W&B disabled, keep local files for reproducibility
        out_parquet = run_dir / "per_sample.parquet"
        df_samples.to_parquet(out_parquet, index=False)
        save_used_cfg(run_dir, script_args, dataset_args, model_args, wandb_args)

    print("Done. Outputs in:", run_dir)


def save_used_cfg(outdir, script_args, dataset_args, model_args, wandb_args):
    cfg = {
        "script": vars(script_args),
        "data": vars(dataset_args),
        "model": vars(model_args),
        "wandb": vars(wandb_args),
    }
    out_file = outdir / "resolved_config.yaml"
    out_file.write_text(
        "\n".join([f"{k}:\n" + "\n".join([f"  {k2}: {v2}" for k2, v2 in v.items()]) for k, v in cfg.items()])
    )

    return out_file


def cli():
    """CLI entry point that handles argument parsing."""
    parser = TrlParser((TopArgs, DatasetArgs, ModelArgs, WandbArgs))
    script_args, dataset_args, model_args, wandb_args = parser.parse_args_and_config()
    main(script_args, dataset_args, model_args, wandb_args)


if __name__ == "__main__":
    cli()
