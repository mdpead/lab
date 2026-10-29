from functools import partial
from typing import Any

import torch
from lab import runs, training
from lab.models.decoder_only import Transformer
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


def train(
    stage: str,
    model: Transformer,
    dataloaders: dict[str, DataLoader],
    tokenizer: PreTrainedTokenizerFast,
    config: runs.Config,
) -> None:
    """Run a pretrain or SFT stage: lab's supervised loop with this module's loss."""
    training.train(
        stage,
        model,
        dataloaders,
        tokenizer,
        config,
        compute_loss=partial(
            compute_loss,
            router_aux_loss_coef=config["stages"][stage]["train"]["router_aux_loss_coef"],
        ),
        tokens_key="input_ids",
    )
