from collections.abc import Mapping
from functools import partial
from typing import Any

import datasets as hf_datasets
import torch
from lab.data import loader_options, pad, token_budget_loaders
from lab.data.samplers import InfiniteRandomSampler, Rows, TokenSampler
from lab.runs import Config
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from src.data import TokenizedDataset


class PromptSampler(TokenSampler):
    """Fixed-size batches of similar-length prompts, endlessly reshuffled.

    A prompt count rather than a token budget: a rollout's cost is dominated by
    max_new_tokens, which is the same for every row. Sorting by length keeps left-padding
    small. The short final batch of a pass is kept, since dropping it would always exclude
    the same longest prompts.
    """

    def __init__(
        self, ds: Rows, prompts_per_batch: int, seed: int, start_minibatch: int = 0
    ) -> None:
        self.prompts_per_batch = prompts_per_batch
        # No token budget: generate_batches below replaces the rule that reads it
        super().__init__(ds, None, seed, start_minibatch)

    def generate_batches(self, ds: Rows) -> list[list[int]]:
        lengths = [row["length"] for row in ds]
        order = sorted(range(len(lengths)), key=lengths.__getitem__)
        return [
            order[i : i + self.prompts_per_batch]
            for i in range(0, len(order), self.prompts_per_batch)
        ]


def collate_rl_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hand the rows straight through: generate() tokenises, and scoring needs the whole row."""
    return batch


def collate_sft_batch(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Tensor]:
    """Pad to the batch maximum and mask the prompt, and any unscored spans, out of the targets.

    Masked positions are overwritten in output_ids only, so the model still reads them.
    """
    ids, padding_mask = pad([row["ids"] for row in batch], pad_token_id)

    input_ids = ids[:, :-1]
    # Cloned, so masking the targets below does not blank the input as well
    output_ids = ids[:, 1:].clone()

    # The target for position p in `ids` sits at p - 1. Trailing pads need no masking.
    for i, row in enumerate(batch):
        output_ids[i, : row["prompt_len"] - 1] = pad_token_id
        for position in row.get("unscored", ()):
            output_ids[i, position - 1] = pad_token_id

    return {
        "input_ids": input_ids,
        "padding_mask": padding_mask[:, :-1],
        "output_ids": output_ids,
    }


def collate_lm_batch(batch: list[Tensor], pad_token_id: int) -> dict[str, Tensor]:
    input_ids = torch.stack(batch)
    padding_mask = (input_ids != pad_token_id).bool()
    return {
        "input_ids": input_ids[:, :-1],
        "padding_mask": padding_mask[:, :-1],
        # CrossEntropyLoss needs Long targets, and the ids are stored int32
        "output_ids": input_ids[:, 1:].long(),
    }


def create_dataloaders_pretrain(
    ds: Mapping[str, TokenizedDataset],
    tokenizer: PreTrainedTokenizerFast,
    config: Config,
) -> dict[str, DataLoader]:

    train_config = config["stages"]["pretrain"]["train"]
    batch_size = train_config["minibatch_token_size"] // config["model"]["max_length"]

    dataloaders = {}
    for split in ds:
        # batch_size lets the sampler convert a minibatch count into a row offset on resume
        sampler = (
            InfiniteRandomSampler(ds[split], config["seed"], batch_size=batch_size)
            if split == "train"
            else None
        )
        dataloaders[split] = DataLoader(
            ds[split],
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=partial(collate_lm_batch, pad_token_id=tokenizer.pad_token_id),
            **loader_options(train_config, split),
        )
    return dataloaders


def create_dataloaders_sft(
    ds: dict[str, hf_datasets.Dataset],
    tokenizer: PreTrainedTokenizerFast,
    config: Config,
) -> dict[str, DataLoader]:

    return token_budget_loaders(
        ds,
        partial(collate_sft_batch, pad_token_id=tokenizer.pad_token_id),
        config["stages"]["sft"]["train"],
        config["seed"],
    )


def create_dataloaders_rl(
    ds: dict[str, list[dict[str, Any]]],
    tokenizer: PreTrainedTokenizerFast,
    config: Config,
) -> dict[str, DataLoader]:
    """Batches of prompt rows for rollout generation.

    No workers or pinned memory: rows are plain dicts and generate() does the tokenising.
    `tokenizer` is unused, kept so the create_dataloaders_* functions share a signature.
    """
    train_config = config["stages"]["rl"]["train"]

    dataloaders: dict[str, DataLoader] = {}
    for split in ds:
        batching: dict[str, Any]
        if split == "train":
            batching = {
                "batch_sampler": PromptSampler(
                    ds[split], train_config["generation_prompts_size"], config["seed"]
                )
            }
        else:
            # One bounded pass in dataset order, so every validation sees the same prompts
            batch_size = train_config.get(
                "validation_prompts_size", train_config["generation_prompts_size"]
            )
            batching = {"batch_size": batch_size, "shuffle": False}

        dataloaders[split] = DataLoader(
            ds[split],
            collate_fn=collate_rl_batch,
            num_workers=0,
            **batching,
        )
    return dataloaders
