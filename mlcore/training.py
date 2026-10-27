import json
import logging
import os
from collections.abc import Callable
from typing import Any, NotRequired, TypedDict

import torch
from torch import amp, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from mlcore import runs
from mlcore.models.decoder_only import Transformer


class Run(TypedDict):
    """A stage's training state, fresh from create_run or resumed by load_run."""

    model: Transformer
    criterion: nn.CrossEntropyLoss
    optimiser: Optimizer
    lr_scheduler: LRScheduler
    scaler: amp.GradScaler
    results: list[dict[str, Any]]
    step_no: int
    run_path: NotRequired[str]


# A stage's training loop, as train() calls it
TrainLoop = Callable[
    [str, Transformer, dict[str, DataLoader], PreTrainedTokenizerFast, Run, runs.Config], None
]


class WarmupInverseSquareRootLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warm_up_steps: int,
        last_epoch: int = -1,
    ) -> None:

        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.warm_up_steps = warm_up_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step_no = self.last_epoch + 1
        if self.last_epoch < self.warm_up_steps:
            lrs = [base_lr * (step_no) / self.warm_up_steps for base_lr in self.base_lrs]
        else:
            lrs = [
                base_lr * (self.warm_up_steps**0.5) / ((step_no) ** 0.5)
                for base_lr in self.base_lrs
            ]
        return lrs


class WarmupConstantLR(LRScheduler):
    """Linear warmup to the base rate, then held there.

    For RL, where the policy keeps generating new data and late steps matter as much.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warm_up_steps: int,
        last_epoch: int = -1,
    ) -> None:

        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.warm_up_steps = warm_up_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        scale = min(1.0, (self.last_epoch + 1) / self.warm_up_steps)
        return [base_lr * scale for base_lr in self.base_lrs]


_LR_SCHEDULES = {
    "inverse_sqrt": WarmupInverseSquareRootLR,
    "constant": WarmupConstantLR,
}


def checkpoint_steps_on_disk(checkpoints_path: str) -> list[int]:
    """Step numbers of the checkpoints in a run, ascending. Filenames are "<step>.pt"."""
    if not os.path.isdir(checkpoints_path):
        return []
    steps = []
    for name in os.listdir(checkpoints_path):
        stem, ext = os.path.splitext(name)
        if ext == ".pt" and stem.isdigit():
            steps.append(int(stem))
    return sorted(steps)


def _prune_checkpoints(checkpoints_path: str, keep: int | None) -> None:
    """Delete all but the `keep` most recent checkpoints, ordered by step number.

    Called after the new checkpoint is written, so an interrupted prune leaves too many.
    """
    if keep is None:
        return
    if keep < 1:
        raise ValueError(f"keep_checkpoints must be >= 1, got {keep}")

    steps = checkpoint_steps_on_disk(checkpoints_path)
    for step in steps[:-keep]:
        os.remove(f"{checkpoints_path}/{step}.pt")
    if len(steps) > keep:
        logging.info(f"Pruned {len(steps) - keep} checkpoint(s), keeping steps {steps[-keep:]}")


def save_checkpoint(
    model: Transformer,
    optimiser: Optimizer,
    lr_scheduler: LRScheduler,
    scaler: amp.GradScaler,
    run_path: str,
    step_no: int,
    results: list[dict[str, Any]],
    keep_checkpoints: int | None = None,
) -> None:
    checkpoints_path = f"{run_path}/checkpoints"
    os.makedirs(checkpoints_path, exist_ok=True)

    unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "model_state_dict": unwrapped.state_dict(),
        "optimizer_state_dict": optimiser.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step_no": step_no,
    }

    torch.save(checkpoint, f"{checkpoints_path}/{step_no}.pt")
    with open(f"{run_path}/results.json", "w") as f:
        json.dump(results, f)
    logging.info(f"Checkpoint saved at step {step_no}")
    _prune_checkpoints(checkpoints_path, keep_checkpoints)
    return None


def load_checkpoint(
    run_path: str, step_no: int, device: torch.device | None = None
) -> dict[str, Any]:
    checkpoints_path = f"{run_path}/checkpoints"
    if device is not None:
        checkpoint = torch.load(f"{checkpoints_path}/{step_no}.pt", map_location=device)
    else:
        checkpoint = torch.load(f"{checkpoints_path}/{step_no}.pt")

    logging.info(f"Checkpoint loaded from step {step_no}")
    return checkpoint


def create_training_objects(
    model: Transformer, train_config: dict[str, Any], tokenizer: PreTrainedTokenizerFast
) -> tuple[nn.CrossEntropyLoss, Optimizer, LRScheduler, amp.GradScaler]:

    device = torch.device(train_config["device"])
    # label_smoothing is defaulted because RL builds these objects and never uses the criterion
    criterion = nn.CrossEntropyLoss(
        reduction="mean",
        label_smoothing=train_config.get("label_smoothing", 0.0),
        ignore_index=tokenizer.pad_token_id,
    ).to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        betas=train_config["adam_betas"],
        eps=train_config["adam_eps"],
        weight_decay=0.01,
    )

    # Defaulted, so stages that predate the key keep their schedule
    schedule = _LR_SCHEDULES[train_config.get("lr_schedule", "inverse_sqrt")]
    lr_scheduler = schedule(optimiser, train_config["warm_up_steps"])

    scaler = amp.GradScaler()

    return criterion, optimiser, lr_scheduler, scaler


def create_run(
    model: Transformer, train_config: dict[str, Any], tokenizer: PreTrainedTokenizerFast
) -> Run:
    results: list[dict[str, Any]] = []

    device = torch.device(train_config["device"])
    model.to(device)

    criterion, optimiser, lr_scheduler, scaler = create_training_objects(
        model, train_config, tokenizer
    )

    return {
        "model": model,
        "criterion": criterion,
        "optimiser": optimiser,
        "lr_scheduler": lr_scheduler,
        "scaler": scaler,
        "results": results,
        "step_no": 0,
    }


def load_run(
    run_path: str,
    model: Transformer,
    train_config: dict[str, Any],
    tokenizer: PreTrainedTokenizerFast,
) -> Run:
    steps = checkpoint_steps_on_disk(f"{run_path}/checkpoints")
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {run_path}")

    checkpoint_latest_step = steps[-1]

    device = torch.device(train_config["device"])
    model.to(device)

    checkpoint = load_checkpoint(run_path, checkpoint_latest_step, device)

    criterion, optimiser, lr_scheduler, scaler = create_training_objects(
        model, train_config, tokenizer
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    step_no = checkpoint["step_no"]
    results_path = f"{run_path}/results.json"
    results = json.load(open(results_path)) if os.path.exists(results_path) else []

    return {
        "model": model,
        "criterion": criterion,
        "optimiser": optimiser,
        "lr_scheduler": lr_scheduler,
        "scaler": scaler,
        "results": results,
        "step_no": step_no,
    }


def load_model_weights(
    model: Transformer, from_path: str, device: torch.device, step: int | None = None
) -> int:
    """Start a stage from another stage's weights: parameters only.

    Not a resume: optimiser, scheduler and step counter are built fresh. `step` pins which
    checkpoint to take; None takes the latest, which leaves the ancestry unrecorded.
    """
    steps = checkpoint_steps_on_disk(f"{from_path}/checkpoints")
    if not steps:
        raise FileNotFoundError(
            f"No checkpoints in {from_path} to initialise from — run that stage first."
        )
    if step is None:
        step = steps[-1]
        logging.warning(
            f"init_from_step not set; taking the latest checkpoint ({step}) from {from_path}. "
            "Pin it in the config so this run's ancestry is recorded."
        )
    elif step not in steps:
        raise FileNotFoundError(
            f"init_from_step {step} not found in {from_path}/checkpoints (have {steps})."
        )

    model.to(device)
    checkpoint = load_checkpoint(from_path, step, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"Initialised model weights from {from_path} step {step}")
    return step


def get_run(
    run_path: str,
    model: Transformer,
    train_config: dict[str, Any],
    tokenizer: PreTrainedTokenizerFast,
    init_from: str | None = None,
) -> Run:
    if checkpoint_steps_on_disk(f"{run_path}/checkpoints"):
        # A stage with its own history resumes from it and ignores init_from
        run = load_run(run_path, model, train_config, tokenizer)
    else:
        if init_from is not None:
            load_model_weights(
                model,
                init_from,
                torch.device(train_config["device"]),
                train_config.get("init_from_step"),
            )
        run = create_run(model, train_config, tokenizer)

    run["run_path"] = run_path
    return run


def train(
    stage: str,
    model: Transformer,
    dataloaders: dict[str, DataLoader],
    tokenizer: PreTrainedTokenizerFast,
    config: runs.Config,
    init_from: str | None = None,
    *,
    loop: TrainLoop,
) -> None:
    """Run a stage. `init_from` is another stage's directory, whose weights seed the first run.

    `loop` is the stage's training loop, supplied by the project, so stages share the
    resume and already-complete handling here.
    """

    train_config = config["train"][stage]
    run_path = runs.get_stage_path(config, stage)
    run = get_run(run_path, model, train_config, tokenizer, init_from)

    if run["step_no"] >= train_config["num_steps"]:
        logging.info("Training already complete.")
        return None

    if train_config.get("compile_model", False):
        model = torch.compile(model, fullgraph=True)

    loop(stage, model, dataloaders, tokenizer, run, config)

    return None
