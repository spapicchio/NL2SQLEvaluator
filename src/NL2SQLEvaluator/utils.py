from dataclasses import field, dataclass
from typing import Literal, List

import wandb


@dataclass
class WandbArgs:
    project: str = field(
        default="text2sql-eval",
        metadata={"help": "Weights & Biases project name for logging"}
    )
    entity: str = field(
        default="spapicchio-politecnico-di-torino",
        metadata={"help": "Weights & Biases entity/username"}
    )
    group: str = field(
        default="evals",
        metadata={"help": "Experiment group name for organizing runs"}
    )
    mode: Literal["online", "offline"] = field(
        default="online",
        metadata={"help": "Logging mode: online (sync to cloud) or offline (local only)"}
    )
    tags: List[str] = field(
        default_factory=lambda: ["eval", "seg"],
        metadata={"help": "List of tags to categorize the experiment run"}
    )
    notes: str = field(
        default="",
        metadata={"help": "Additional notes or description for the experiment"}
    )
    job_type: str = field(
        default="eval",
        metadata={"help": "Type of job being run (e.g., eval, train, test)"}
    )


def utils_init_wandb(wandb_args: WandbArgs, run_name: str) -> wandb.sdk.wandb_run.Run | None:
    """Initialize W&B if enabled; return the run or None."""
    mode = wandb_args.mode
    if mode == "disabled":
        return None
    run = wandb.init(
        project=wandb_args.project,
        entity=wandb_args.entity,
        mode=mode,  # "online" | "offline"
        group=wandb_args.group,  # group runs in W&B
        job_type=wandb_args.job_type,
        name=run_name,
        tags=wandb_args.tags,
        notes=wandb_args.notes,
    )
    return run
