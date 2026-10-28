from functools import partial
from typing import Any

import sacrebleu
import torch
from mlcore import runs, training
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from src import generation

# The single training stage: names the config sections and the directory under the run
STAGE = "translation"


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


def compute_loss(
    model: nn.Module, batch: dict[str, Any], criterion: nn.Module
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = model(
        batch["src_input_ids"],
        batch["tgt_input_ids"],
        batch["src_padding_mask"],
        batch["tgt_padding_mask"],
    )
    loss = criterion(logits.reshape(-1, logits.shape[2]), batch["tgt_output_ids"].reshape(-1))
    return loss, {"loss": loss}


def train_loop(
    stage: str,
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    tokenizer: PreTrainedTokenizerFast,
    run: training.Run,
    config: runs.Config,
) -> None:
    device = torch.device(config["train"][stage]["device"])
    max_length = config["model"]["max_length"]
    training.supervised_loop(
        stage,
        model,
        dataloaders,
        tokenizer,
        run,
        config,
        compute_loss=partial(compute_loss, criterion=run["criterion"]),
        tokens_key="src_input_ids",
        validation_metrics=lambda m, batch: generate_metrics(
            m, batch, max_length, tokenizer, device
        ),
    )


def train(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    tokenizer: PreTrainedTokenizerFast,
    config: runs.Config,
) -> None:
    """Train the translation stage: training.train's bookkeeping, this module's loop."""
    return training.train(STAGE, model, dataloaders, tokenizer, config, loop=train_loop)
