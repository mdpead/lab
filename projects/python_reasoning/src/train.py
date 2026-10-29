from functools import partial
from typing import Any

import torch
from mlcore import runs, training
from mlcore.models.decoder_only import Transformer
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast


def compute_loss(
    model: Transformer, batch: dict[str, Any], criterion: nn.Module, router_aux_loss_coef: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # The mask keeps pad tokens out of the MoE's seat assignment
    logits, loss_aux = model(batch["input_ids"], padding_mask=batch["padding_mask"])
    loss_ce = criterion(logits.reshape(-1, logits.shape[2]), batch["output_ids"].reshape(-1))
    loss = loss_ce + loss_aux * router_aux_loss_coef
    return loss, {"loss_ce": loss_ce, "loss_aux": loss_aux}


def train_loop(
    stage: str,
    model: Transformer,
    dataloaders: dict[str, DataLoader],
    tokenizer: PreTrainedTokenizerFast,
    run: training.Run,
    config: runs.Config,
) -> None:
    training.supervised_loop(
        stage,
        model,
        dataloaders,
        tokenizer,
        run,
        config,
        compute_loss=partial(
            compute_loss,
            criterion=run["criterion"],
            router_aux_loss_coef=config["train"][stage]["router_aux_loss_coef"],
        ),
        tokens_key="input_ids",
    )


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
