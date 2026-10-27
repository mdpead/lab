import itertools

import torch

from src import dataloader


def test_token_sampler_respects_budget_and_covers_all_rows(ds_tokenized):
    ds = ds_tokenized["train"]
    budget = 30
    sampler = dataloader.TokenSampler(ds, budget, seed=0)
    lengths = {
        i: max(en, cy) for i, en, cy in zip(ds["idx"], ds["en_token_length"], ds["cy_token_length"])
    }

    for batch in sampler.batches:
        assert len(batch) == 1 or sum(lengths[i] for i in batch) <= budget

    flat = [i for batch in sampler.batches for i in batch]
    assert sorted(flat) == sorted(ds["idx"])
    assert len(sampler) == len(sampler.batches)


def test_token_sampler_oversized_row_gets_own_batch(ds_tokenized):
    sampler = dataloader.TokenSampler(ds_tokenized["train"], 1, seed=0)
    assert all(len(batch) == 1 for batch in sampler.batches)


def test_token_sampler_iterates_forever_and_is_seeded(ds_tokenized):
    ds = ds_tokenized["train"]
    n = len(dataloader.TokenSampler(ds, 30, seed=0))

    def take(seed):
        return list(itertools.islice(iter(dataloader.TokenSampler(ds, 30, seed=seed)), 3 * n))

    assert take(0) == take(0)
    assert take(0) != take(1)
    first_epoch = take(0)[:n]
    assert sorted(map(tuple, first_epoch)) == sorted(
        map(tuple, dataloader.TokenSampler(ds, 30, seed=0).batches)
    )


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
    loaders = dataloader.create_dataloaders(ds_tokenized, config)
    assert set(loaders) == {"train", "test"}
    batch = next(iter(loaders["train"]))
    assert (
        batch["src_input_ids"].shape[0] == batch["tgt_input_ids"].shape[0] == len(batch["src_text"])
    )
    assert batch["tgt_input_ids"].shape == batch["tgt_output_ids"].shape
