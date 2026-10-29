import itertools
import json
from pathlib import Path

import torch
from lab.testing import batch_digest

from datasets import Dataset, DatasetDict
from src import dataloader
from src.train import STAGE


def test_batches_fit_the_padded_token_budget(ds_tokenized, tok, config):
    budget = config["stages"][STAGE]["train"]["minibatch_token_size"]
    loader = dataloader.create_dataloaders(ds_tokenized, tok, config, STAGE)["train"]

    for batch in itertools.islice(loader, 20):
        rows = batch["src_input_ids"].shape[0]
        # Each side pads to its own longest row; the batch is costed by the longer side
        longest = max(batch["src_input_ids"].shape[1], batch["tgt_input_ids"].shape[1] + 1)
        assert rows == 1 or rows * longest <= budget


def test_collate_batch_pads_and_shifts_target():
    pad = 2
    batch = [
        {
            "text_en_tokenized": [0, 5, 6, 1],
            "text_cy_tokenized": [0, 7, 1],
            "text_en": "a b",
            "text_cy": "c",
        },
        {
            "text_en_tokenized": [0, 8, 1],
            "text_cy_tokenized": [0, 9, 10, 11, 1],
            "text_en": "d",
            "text_cy": "e f g",
        },
    ]
    out = dataloader.collate_batch(batch, pad)

    assert out["src_input_ids"].tolist() == [[0, 5, 6, 1], [0, 8, 1, pad]]
    assert out["src_padding_mask"].tolist() == [[True] * 4, [True, True, True, False]]
    assert out["tgt_input_ids"].tolist() == [[0, 7, 1, pad], [0, 9, 10, 11]]
    assert out["tgt_output_ids"].tolist() == [[7, 1, pad, pad], [9, 10, 11, 1]]
    assert out["tgt_padding_mask"].tolist() == [[True, True, True, False], [True] * 4]
    assert out["src_text"] == ["a b", "d"]
    assert out["tgt_text"] == ["c", "e f g"]
    assert out["src_input_ids"].dtype == torch.long


def test_create_dataloaders(ds_tokenized, tok, config):
    loaders = dataloader.create_dataloaders(ds_tokenized, tok, config, STAGE)
    assert set(loaders) == {"train", "test"}
    batch = next(iter(loaders["train"]))
    assert (
        batch["src_input_ids"].shape[0] == batch["tgt_input_ids"].shape[0] == len(batch["src_text"])
    )
    assert batch["tgt_input_ids"].shape == batch["tgt_output_ids"].shape


# --- batches recorded from the loaders before they moved into lab ---

REFERENCE = json.loads((Path(__file__).parent / "fixtures/batches.json").read_text())


def _rows(n, offset):
    """Ragged rows with distinct ids, so a reordered or mis-padded batch changes the hash."""
    rows = []
    for r in range(n):
        en = [0, *range(offset + 10 * r, offset + 10 * r + 2 + r % 5), 1]
        cy = [0, *range(offset + 10 * r + 5, offset + 10 * r + 6 + r % 3), 1]
        rows.append(
            {
                "text_en": f"en {r}",
                "text_cy": f"cy {r}",
                "text_en_tokenized": en,
                "text_cy_tokenized": cy,
                "length": max(len(en), len(cy)),
            }
        )
    return Dataset.from_list(rows)


def test_loaders_yield_the_recorded_batches(tok, config):
    config["stages"][STAGE]["train"]["num_workers"] = 0
    ds = DatasetDict(train=_rows(40, 100), test=_rows(12, 1000))
    loaders = dataloader.create_dataloaders(ds, tok, config, STAGE)
    digests = {
        "train": batch_digest(loaders["train"], 30),
        "test": batch_digest(loaders["test"], 30),
    }
    assert digests == REFERENCE
