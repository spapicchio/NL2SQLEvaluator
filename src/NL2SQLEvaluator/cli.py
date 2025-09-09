# args.py
import pathlib
from time import gmtime, strftime

import wandb
from dotenv import load_dotenv

from NL2SQLEvaluator.hf_argument_parser import TrlParser
from NL2SQLEvaluator.logger import get_logger
from NL2SQLEvaluator.main_run_evaluation import run_evaluation, ScriptArgs, DatasetArgs, ModelArgs
from NL2SQLEvaluator.utils_wandb import utils_init_wandb, WandbArgs

load_dotenv(override=True)


def _main(script_args, dataset_args, model_args, wandb_args):
    # Init W&B (respect wandb.mode: online|offline|disabled)
    wandb_run = utils_init_wandb(wandb_args, run_name=f"eval__{model_args.model_name}__{dataset_args.dataset_name}")

    # --- run evaluation ---
    summary, df_samples = run_evaluation(script_args, dataset_args, model_args)

    # --- Store Results ---
    run_dir = pathlib.Path(script_args.output_dir)
    today = str(strftime("%Y-%m-%d", gmtime()))
    hours_minutes = str(strftime("%Hh-%Mm", gmtime()))
    run_dir = run_dir / today / hours_minutes / f"{model_args.model_name}_{dataset_args.dataset_name}_s{script_args.seed}"
    # if run_dir does not exist create it
    run_dir.mkdir(parents=True, exist_ok=True)

    if wandb_run is not None:
        table = wandb.Table(columns=list(df_samples.columns))
        for row in df_samples.itertuples(index=False):
            # transform list element in row into str
            row = tuple(str(x) if not isinstance(x, str) else x for x in row)
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
        cfg_path = _save_used_cfg(run_dir, script_args, dataset_args, model_args, wandb_args)
        cfg_art = wandb.Artifact(name=f"config_wandb_id_{wandb_run.id}", type="config")
        cfg_art.add_file(str(cfg_path))
        wandb_run.log_artifact(cfg_art)
        wandb_run.finish()
    else:
        # Even with W&B disabled, keep local files for reproducibility
        out_parquet = run_dir / "per_sample.parquet"
        df_samples.to_parquet(out_parquet, index=False)
        _save_used_cfg(run_dir, script_args, dataset_args, model_args, wandb_args)

    get_logger(name='main').info(f"Done. Outputs in: {run_dir}")


def _save_used_cfg(outdir, script_args, dataset_args, model_args, wandb_args):
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
    parser = TrlParser((ScriptArgs, DatasetArgs, ModelArgs, WandbArgs))
    script_args, dataset_args, model_args, wandb_args = parser.parse_args_and_config()
    _main(script_args, dataset_args, model_args, wandb_args)


if __name__ == "__main__":
    cli()
