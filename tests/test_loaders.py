"""Padding ragged rows, and the loader settings each split gets."""

import torch

from lab.data import loader_options, pad, token_budget_loaders


def test_pad_right_pads_to_the_longest_row_and_masks_the_padding():
    ids, mask = pad([[5, 6, 7], [8]], pad_id=0)
    assert ids.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert ids.dtype == torch.long
    assert mask.tolist() == [[True, True, True], [True, False, False]]


def test_only_the_train_split_gets_workers():
    train_config = {"num_workers": 4, "prefetch_factor": 3}
    assert loader_options(train_config, "train") == {
        "pin_memory": True,
        "num_workers": 4,
        "prefetch_factor": 3,
    }
    assert loader_options(train_config, "test")["num_workers"] == 0


def test_prefetch_factor_is_dropped_without_workers():
    """DataLoader rejects a prefetch_factor alongside num_workers=0."""
    assert loader_options({"prefetch_factor": 3}, "train")["prefetch_factor"] is None


def test_token_budget_loaders_batch_each_split_within_the_budget():
    rows = [{"ids": list(range(n)), "length": n} for n in (2, 3, 3, 5, 8, 8)]
    loaders = token_budget_loaders(
        {"train": rows, "test": rows[:3]},
        collate_fn=lambda batch: [len(row["ids"]) for row in batch],
        train_config={"minibatch_token_size": 10},
        seed=0,
    )
    assert set(loaders) == {"train", "test"}
    for lengths in [next(iter(loaders["train"])) for _ in range(6)]:
        assert len(lengths) == 1 or len(lengths) * max(lengths) <= 10
