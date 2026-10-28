import itertools
import logging
import time
from collections.abc import Iterable
from typing import Any

import sacrebleu
import torch
from mlcore import runs, training
from mlcore.data.samplers import resume_at_minibatch
from torch import amp, nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from src import generation

# The single training stage: names the config sections and the directory under the run
STAGE = "translation"

MODEL_INPUTS = [
    "src_input_ids",
    "tgt_input_ids",
    "src_padding_mask",
    "tgt_padding_mask",
    "tgt_output_ids",
]


def generate_metrics(
    model: nn.Module,
    batch: dict[str, Any],
    max_length: int,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> dict[str, float]:

    pred_text = generation.generate_texts(
        model, tokenizer, input_texts=batch["src_text"], max_length=max_length, device=device
    )

    sacrebleu_score = sacrebleu.corpus_bleu(
        pred_text,
        [batch["tgt_text"]],
        smooth_method="exp",
        smooth_value=0.0,
        lowercase=False,
        tokenize="intl",
    )

    metrics = {
        "bleu": sacrebleu_score.score,
    }

    return metrics


def validation_step(
    model: nn.Module,
    dataloader: Iterable[dict[str, Any]],
    criterion: nn.Module,
    device: torch.device,
    tokenizer: PreTrainedTokenizerBase,
    step_no: int,
    max_length: int,
    validation_minibatches: int,
) -> dict[str, Any]:

    model.eval()
    start_time = time.time()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    bleu_batch = None

    for minibatch in itertools.islice(dataloader, validation_minibatches):
        minibatch = {
            k: v.to(device, non_blocking=True) if k in MODEL_INPUTS else v
            for k, v in minibatch.items()
        }

        with torch.no_grad():
            logits = model(
                minibatch["src_input_ids"],
                minibatch["tgt_input_ids"],
                minibatch["src_padding_mask"],
                minibatch["tgt_padding_mask"],
            )
            loss = criterion(
                logits.reshape(-1, logits.shape[2]), minibatch["tgt_output_ids"].reshape(-1)
            )

        total_loss += loss.item()
        total_tokens += minibatch["src_input_ids"].ne(tokenizer.pad_token_id).sum().item()
        num_batches += 1
        if bleu_batch is None:
            bleu_batch = minibatch

    if bleu_batch is None:
        raise ValueError("Validation dataloader yielded no batches")

    elapsed_time = time.time() - start_time
    validation_metrics = generate_metrics(model, bleu_batch, max_length, tokenizer, device)

    return {
        "type": "validation",
        "step_no": step_no,
        "num_tokens": total_tokens,
        "tokens_per_sec": total_tokens / elapsed_time,
        "loss": total_loss / num_batches,
        "validation_metrics": validation_metrics,
    }


def train_loop(
    stage: str,
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    tokenizer: PreTrainedTokenizerFast,
    run: training.Run,
    config: runs.Config,
) -> None:

    train_config = config["train"][stage]
    device = torch.device(train_config["device"])
    grad_accum_steps = (
        train_config["effective_batch_token_size"] // train_config["minibatch_token_size"]
    )
    num_steps = train_config["num_steps"]
    checkpoint_steps = train_config["checkpoint_steps"]
    keep_checkpoints = train_config.get("keep_checkpoints")
    validation_steps = train_config["validation_steps"]
    validation_batches = train_config["validation_batches"]
    max_length = config["model"]["max_length"]
    cache_clear_steps = train_config.get("cache_clear_steps")

    criterion = run["criterion"]
    optimiser = run["optimiser"]
    lr_scheduler = run["lr_scheduler"]
    scaler = run["scaler"]
    results = run["results"]
    step_no = run["step_no"]
    run_path = run["run_path"]

    batch_tokens = 0
    start_time = time.time()
    total_loss = 0.0
    # Resume where we left off by skipping the sampler forward, counted in minibatches
    resume_at_minibatch(dataloaders["train"], step_no * grad_accum_steps)
    optimiser.zero_grad(set_to_none=True)

    model.train()
    for accum_idx, batch in enumerate(dataloaders["train"]):
        # Move batch to device
        batch = {
            k: v.to(device, non_blocking=True) if k in MODEL_INPUTS else v for k, v in batch.items()
        }

        with amp.autocast(device_type=device.type):
            logits = model(
                batch["src_input_ids"],
                batch["tgt_input_ids"],
                batch["src_padding_mask"],
                batch["tgt_padding_mask"],
            )
            loss = criterion(
                logits.reshape(-1, logits.shape[2]), batch["tgt_output_ids"].reshape(-1)
            )

        minibatch_tokens = batch["src_input_ids"].ne(tokenizer.pad_token_id).sum().item()
        batch_tokens += minibatch_tokens

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()
        total_loss += scaled_loss.item()

        # Gradient accumulation
        if (accum_idx + 1) % grad_accum_steps != 0:
            continue

        # Step optimiser and scheduler
        scaler.unscale_(optimiser)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()
        lr_scheduler.step()
        optimiser.zero_grad(set_to_none=True)

        step_no += 1

        elapsed_time = time.time() - start_time
        result: dict[str, Any] = {}
        result["type"] = "train"
        result["step_no"] = step_no
        result["num_tokens"] = batch_tokens
        result["tokens_per_sec"] = batch_tokens / elapsed_time
        result["learning_rate"] = lr_scheduler.get_last_lr()[0]
        result["loss"] = total_loss
        result["token_length"] = batch["src_input_ids"].shape[1]
        result["grad_norm"] = grad_norm.item()
        results.append(result)
        logging.info(result)

        # Validation step
        if step_no % validation_steps == 0:
            validation_result = validation_step(
                model,
                dataloaders["test"],
                criterion,
                device,
                tokenizer,
                step_no,
                max_length,
                validation_batches * grad_accum_steps,
            )
            logging.info(validation_result)
            results.append(validation_result)
            model.train()

        # Checkpointing
        if step_no % checkpoint_steps == 0:
            training.save_checkpoint(
                model, optimiser, lr_scheduler, scaler, run_path, step_no, results, keep_checkpoints
            )

        # Delete tensors to free up memory
        del batch, logits, loss, scaled_loss
        if cache_clear_steps and step_no % cache_clear_steps == 0:
            torch.cuda.empty_cache()

        # Reset counters
        batch_tokens = 0
        start_time = time.time()
        total_loss = 0.0

        # Stop after num_steps
        if step_no >= num_steps:
            break

    # Final save when num_steps isn't a multiple of checkpoint_steps, so the tail isn't lost.
    if step_no % checkpoint_steps != 0:
        training.save_checkpoint(
            model, optimiser, lr_scheduler, scaler, run_path, step_no, results, keep_checkpoints
        )

    return None


def train(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    tokenizer: PreTrainedTokenizerFast,
    config: runs.Config,
) -> None:
    """Train the translation stage: training.train's bookkeeping, this module's loop."""
    return training.train(STAGE, model, dataloaders, tokenizer, config, loop=train_loop)
