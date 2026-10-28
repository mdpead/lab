import json
import os

import pytest
import torch
from mlcore import runs, training

from src import dataloader, train
from src.train import STAGE


def test_generate_metrics_perfect_and_wrong(monkeypatch, tok):
    batch = {"src_text": ["Good morning"], "tgt_text": ["Mae'r tywydd yn braf heddiw"]}

    monkeypatch.setattr(
        train.generation, "generate_texts", lambda *a, **k: ["Mae'r tywydd yn braf heddiw"]
    )
    assert train.generate_metrics(None, batch, 32, tok, "cpu")["bleu"] == pytest.approx(100.0)

    monkeypatch.setattr(train.generation, "generate_texts", lambda *a, **k: ["Ble mae"])
    assert train.generate_metrics(None, batch, 32, tok, "cpu")["bleu"] == pytest.approx(0.0)


def test_validation_step_rejects_empty_dataloader(transformer, tok):
    with pytest.raises(ValueError, match="no batches"):
        train.validation_step(
            transformer, [], torch.nn.CrossEntropyLoss(), torch.device("cpu"), tok, 0, 32, 1
        )


def run_training(config, transformer, ds_tokenized, tok):
    runs.init_run(config, STAGE)
    loaders = dataloader.create_dataloaders(ds_tokenized, config, STAGE)
    train.train(transformer, loaders, tok, config)
    return runs.get_stage_path(config, STAGE)


def read_results(stage_path):
    return json.load(open(f"{stage_path}/results.json"))


def test_train_end_to_end(config, transformer, ds_tokenized, tok):
    stage_path = run_training(config, transformer, ds_tokenized, tok)

    assert sorted(os.listdir(f"{stage_path}/checkpoints")) == ["2.pt", "4.pt"]
    results = read_results(stage_path)
    train_results = [r for r in results if r["type"] == "train"]
    validation_results = [r for r in results if r["type"] == "validation"]
    assert [r["step_no"] for r in train_results] == [1, 2, 3, 4]
    assert [r["step_no"] for r in validation_results] == [2, 4]
    assert all(r["loss"] > 0 and r["num_tokens"] > 0 for r in train_results)
    assert all("bleu" in r["validation_metrics"] for r in validation_results)
    learning_rate = config["train"][STAGE]["learning_rate"]
    assert train_results[0]["learning_rate"] == pytest.approx(learning_rate)


def test_train_resumes_from_checkpoint(config, transformer, ds_tokenized, tok):
    stage_path = run_training(config, transformer, ds_tokenized, tok)

    config["train"][STAGE]["num_steps"] = 6
    run = training.get_run(stage_path, transformer, config["train"][STAGE], tok)
    assert run["step_no"] == 4
    assert len(run["results"]) == 6

    run_training(config, transformer, ds_tokenized, tok)
    results = read_results(stage_path)
    assert [r["step_no"] for r in results if r["type"] == "train"] == [1, 2, 3, 4, 5, 6]
    assert sorted(os.listdir(f"{stage_path}/checkpoints")) == ["2.pt", "4.pt", "6.pt"]


def test_resume_continues_the_same_batch_order(config, ds_tokenized, tok, transformer):
    """Stopping and resuming sees the same minibatches as one uninterrupted run."""
    straight = dict(config, name="straight")
    straight["train"] = {STAGE: dict(config["train"][STAGE], num_steps=4)}
    resumed = dict(config, name="resumed")
    resumed["train"] = {STAGE: dict(config["train"][STAGE], num_steps=2)}

    def tokens_per_step(cfg):
        return [
            r["num_tokens"]
            for r in read_results(runs.get_stage_path(cfg, STAGE))
            if r["type"] == "train"
        ]

    torch.manual_seed(0)
    run_training(straight, transformer, ds_tokenized, tok)
    run_training(resumed, transformer, ds_tokenized, tok)
    resumed["train"][STAGE]["num_steps"] = 4
    run_training(resumed, transformer, ds_tokenized, tok)

    assert tokens_per_step(resumed) == tokens_per_step(straight)


def test_final_step_is_checkpointed(config, transformer, ds_tokenized, tok):
    config["train"][STAGE]["num_steps"] = 3
    stage_path = run_training(config, transformer, ds_tokenized, tok)
    assert sorted(os.listdir(f"{stage_path}/checkpoints")) == ["2.pt", "3.pt"]


def test_train_skips_when_already_complete(config, transformer, ds_tokenized, tok, monkeypatch):
    stage_path = run_training(config, transformer, ds_tokenized, tok)

    def fail(*args, **kwargs):
        raise AssertionError("train_loop should not run")

    monkeypatch.setattr(train, "train_loop", fail)
    run_training(config, transformer, ds_tokenized, tok)
    assert [r["step_no"] for r in read_results(stage_path) if r["type"] == "train"] == [1, 2, 3, 4]
