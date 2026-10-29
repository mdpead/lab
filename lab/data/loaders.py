"""Building a supervised stage's data loaders: padding ragged rows, and each split's loader."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from lab.data.samplers import Rows, TokenSampler


def pad(rows: Sequence[Sequence[int]], pad_id: int) -> tuple[Tensor, Tensor]:
    """Right-pad id lists to the longest: the ids, and a mask that is true on the real tokens."""
    ids = torch.full((len(rows), max(len(row) for row in rows)), pad_id, dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return ids, ids != pad_id


def loader_options(train_config: Mapping[str, Any], split: str) -> dict[str, Any]:
    """The DataLoader settings for a split. Only the train split gets workers: validation is a
    handful of batches, and starting workers for it would cost more than it saves."""
    num_workers = train_config.get("num_workers", 0) if split == "train" else 0
    return {
        "pin_memory": True,
        "num_workers": num_workers,
        # DataLoader rejects a prefetch_factor without workers
        "prefetch_factor": train_config.get("prefetch_factor", 2) if num_workers else None,
    }


def token_budget_loaders(
    ds: Mapping[str, Rows],
    collate_fn: Callable[[list[Any]], Any],
    train_config: Mapping[str, Any],
    seed: int,
) -> dict[str, DataLoader]:
    """A loader per split, batching rows of similar length up to `minibatch_token_size`."""
    return {
        split: DataLoader(
            rows,
            batch_sampler=TokenSampler(rows, train_config["minibatch_token_size"], seed),
            collate_fn=collate_fn,
            **loader_options(train_config, split),
        )
        for split, rows in ds.items()
    }
