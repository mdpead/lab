"""The shared supervised loop, driven by a toy model on CPU."""

import json
from functools import partial

import pytest
import torch
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader

from lab import runs, training
from lab.data.samplers import TokenSampler

STAGE = "s"
PAD = 2


def _config(tmp_path, num_steps=3):
    return {
        "name": "t",
        "seed": 0,
        "locations": {"models_dir": str(tmp_path)},
        "stages": {
            STAGE: {
                "train": {
                    "device": "cpu",
                    "learning_rate": 1e-3,
                    "adam_betas": [0.9, 0.98],
                    "adam_eps": 1e-9,
                    "warm_up_steps": 2,
                    "effective_batch_token_size": 40,
                    "minibatch_token_size": 20,
                    "num_steps": num_steps,
                    "checkpoint_steps": 2,
                    "validation_steps": 2,
                    "validation_batches": 1,
                }
            }
        },
    }


def _collate(rows):
    width = max(len(r["ids"]) for r in rows)
    ids = torch.tensor([r["ids"] + [PAD] * (width - len(r["ids"])) for r in rows])
    return {"input_ids": ids[:, :-1], "output_ids": ids[:, 1:], "text": ["x"] * len(rows)}


def _dataloaders():
    ids = [[0] + [5 + i % 7] * (3 + i % 4) + [1] for i in range(24)]
    ds = Dataset.from_dict({"ids": ids, "length": [len(r) for r in ids]})
    return {
        split: DataLoader(ds, batch_sampler=TokenSampler(ds, 20, seed=0), collate_fn=_collate)
        for split in ("train", "test")
    }


def _compute_loss(model, batch, criterion):
    logits = model(batch["input_ids"])
    loss = criterion(logits.reshape(-1, logits.shape[-1]), batch["output_ids"].reshape(-1))
    return loss, {"loss": loss, "constant": torch.tensor(2.0)}


def _train(tmp_path, tokenizer, num_steps=3, validation_metrics=None):
    config = _config(tmp_path, num_steps)
    model = nn.Sequential(nn.Embedding(16, 8), nn.Linear(8, 16))
    run = training.create_run(model, config["stages"][STAGE]["train"], tokenizer)
    run["run_path"] = runs.get_stage_path(config, STAGE)
    training.supervised_loop(
        STAGE,
        model,
        _dataloaders(),
        tokenizer,
        run,
        config,
        compute_loss=partial(_compute_loss, criterion=run["criterion"]),
        tokens_key="input_ids",
        validation_metrics=validation_metrics,
    )
    return run["run_path"]


def test_logged_values_are_averaged_per_step(tmp_path, tokenizer):
    run_path = _train(tmp_path, tokenizer)
    results = json.load(open(f"{run_path}/results.json"))

    assert [(r["type"], r["step_no"]) for r in results] == [
        ("train", 1),
        ("train", 2),
        ("validation", 2),
        ("train", 3),
    ]
    assert all(r["constant"] == pytest.approx(2.0) for r in results)
    assert all(r["loss"] > 0 for r in results)
    assert "validation_metrics" not in results[2]


def test_validation_metrics_come_from_one_batch(tmp_path, tokenizer):
    seen = []

    def metrics(model, batch):
        seen.append(batch["text"])
        return {"score": 1.0}

    run_path = _train(tmp_path, tokenizer, validation_metrics=metrics)
    validation = [
        r for r in json.load(open(f"{run_path}/results.json")) if r["type"] == "validation"
    ]

    assert len(seen) == 1
    assert validation[0]["validation_metrics"] == {"score": 1.0}


def test_final_step_is_checkpointed(tmp_path, tokenizer):
    run_path = _train(tmp_path, tokenizer, num_steps=3)
    assert training.checkpoint_steps_on_disk(f"{run_path}/checkpoints") == [2, 3]


def test_validation_step_rejects_empty_dataloader():
    with pytest.raises(ValueError, match="no batches"):
        training.validation_step(nn.Linear(2, 2), [], None, "input_ids", PAD, "cpu", 0, 1)
