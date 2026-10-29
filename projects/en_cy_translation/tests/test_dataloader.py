import itertools

import torch

from src import dataloader
from src.train import STAGE


def test_batches_fit_the_padded_token_budget(ds_tokenized, config):
    budget = config["stages"][STAGE]["train"]["minibatch_token_size"]
    loader = dataloader.create_dataloaders(ds_tokenized, config, STAGE)["train"]

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


def test_make_collate_fn(ds_tokenized):
    rows = [ds_tokenized["train"][i] for i in range(3)]
    expected = dataloader.collate_batch(rows, 2)
    actual = dataloader.make_collate_fn(2)(rows)
    assert torch.equal(actual["src_input_ids"], expected["src_input_ids"])


def test_create_dataloaders(ds_tokenized, config):
    loaders = dataloader.create_dataloaders(ds_tokenized, config, STAGE)
    assert set(loaders) == {"train", "test"}
    batch = next(iter(loaders["train"]))
    assert (
        batch["src_input_ids"].shape[0] == batch["tgt_input_ids"].shape[0] == len(batch["src_text"])
    )
    assert batch["tgt_input_ids"].shape == batch["tgt_output_ids"].shape
