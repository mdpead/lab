import itertools
import logging
import time
from typing import Any

import torch
from mlcore import runs, training
from mlcore.data.samplers import resume_at_minibatch
from mlcore.models.decoder_only import Transformer
from torch import amp, nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast


def validation_step(
    model: Transformer,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tokenizer: PreTrainedTokenizerFast,
    step_no: int,
    max_length: int,
    validation_minibatches: int,
) -> dict[str, Any]:

    model.eval()
    start_time = time.time()

    total_loss_ce = 0.0
    total_loss_aux = 0.0
    total_tokens = 0
    num_batches = 0

    for minibatch in itertools.islice(dataloader, validation_minibatches):
        minibatch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in minibatch.items()
        }

        with torch.no_grad():
            logits, loss_aux = model(minibatch["input_ids"], padding_mask=minibatch["padding_mask"])
            loss = criterion(
                logits.reshape(-1, logits.shape[2]), minibatch["output_ids"].reshape(-1)
            )

        total_loss_ce += loss.item()
        total_loss_aux += loss_aux.item()
        total_tokens += minibatch["input_ids"].ne(tokenizer.pad_token_id).sum().item()
        num_batches += 1

    elapsed_time = time.time() - start_time

    return {
        "type": "validation",
        "step_no": step_no,
        "num_tokens": total_tokens,
        "tokens_per_sec": total_tokens / elapsed_time,
        "loss_ce": total_loss_ce / num_batches,
        "loss_aux": total_loss_aux / num_batches,
    }


def train_loop(
    stage: str,
    model: Transformer,
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
    router_aux_loss_coef = train_config["router_aux_loss_coef"]

    criterion = run["criterion"]
    optimiser = run["optimiser"]
    lr_scheduler = run["lr_scheduler"]
    scaler = run["scaler"]
    results = run["results"]
    step_no = run["step_no"]
    run_path = run["run_path"]

    batch_tokens = 0
    start_time = time.time()
    total_loss_ce = 0.0
    total_loss_aux = 0.0
    # Resume where we left off by skipping the sampler forward, counted in minibatches
    resume_at_minibatch(dataloaders["train"], step_no * grad_accum_steps)
    optimiser.zero_grad(set_to_none=True)

    model.train()
    for accum_idx, batch in enumerate(dataloaders["train"]):
        # Move batch to device
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with amp.autocast(device_type=device.type):
            # The mask keeps pad tokens out of the MoE's seat assignment
            logits, loss_aux = model(batch["input_ids"], padding_mask=batch["padding_mask"])
            loss_ce = criterion(
                logits.reshape(-1, logits.shape[2]), batch["output_ids"].reshape(-1)
            )
            loss = loss_ce + loss_aux * router_aux_loss_coef

        minibatch_tokens = batch["input_ids"].ne(tokenizer.pad_token_id).sum().item()
        batch_tokens += minibatch_tokens

        scaled_loss = loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()
        total_loss_ce += loss_ce.item() / grad_accum_steps
        total_loss_aux += loss_aux.item() / grad_accum_steps

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
        result["loss_ce"] = total_loss_ce
        result["loss_aux"] = total_loss_aux
        result["token_length"] = batch["input_ids"].shape[1]
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
        del batch, logits, loss, loss_ce, loss_aux, scaled_loss
        if cache_clear_steps and step_no % cache_clear_steps == 0:
            torch.cuda.empty_cache()

        # Reset counters
        batch_tokens = 0
        start_time = time.time()
        total_loss_ce = 0.0
        total_loss_aux = 0.0

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
    stage: str,
    model: Transformer,
    dataloaders: dict[str, DataLoader],
    tokenizer: PreTrainedTokenizerFast,
    config: runs.Config,
    init_from: str | None = None,
) -> None:
    """Run a pretrain or SFT stage: training.train's bookkeeping, this module's loop."""
    return training.train(stage, model, dataloaders, tokenizer, config, init_from, loop=train_loop)
