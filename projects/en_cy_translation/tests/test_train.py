import json
import os
from pathlib import Path

import pytest
import torch
from lab import runs, training
from lab.models import build_model
from lab.tokenizer import train_tokenizer

from src import dataloader, tokenizer, train
from src.config import Config
from src.train import STAGE


def test_generate_metrics_perfect_and_wrong(monkeypatch, tok):
    batch = {"src_text": ["Good morning"], "tgt_text": ["Mae'r tywydd yn braf heddiw"]}

    monkeypatch.setattr(
        train.generation, "generate_texts", lambda *a, **k: ["Mae'r tywydd yn braf heddiw"]
    )
    assert train.generate_metrics(None, batch, 32, tok, "cpu")["bleu"] == pytest.approx(100.0)

    monkeypatch.setattr(train.generation, "generate_texts", lambda *a, **k: ["Ble mae"])
    assert train.generate_metrics(None, batch, 32, tok, "cpu")["bleu"] == pytest.approx(0.0)


def run_training(config, transformer, ds_tokenized, tok):
    runs.init_run(config, STAGE, Config)
    loaders = dataloader.create_dataloaders(ds_tokenized, tok, config, STAGE)
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
    learning_rate = config["stages"][STAGE]["train"]["learning_rate"]
    assert train_results[0]["learning_rate"] == pytest.approx(learning_rate)


def test_train_resumes_from_checkpoint(config, transformer, ds_tokenized, tok):
    stage_path = run_training(config, transformer, ds_tokenized, tok)

    config["stages"][STAGE]["train"]["num_steps"] = 6
    run = training.get_run(stage_path, transformer, config["stages"][STAGE]["train"], tok)
    assert run["step_no"] == 4
    assert len(run["results"]) == 6

    run_training(config, transformer, ds_tokenized, tok)
    results = read_results(stage_path)
    assert [r["step_no"] for r in results if r["type"] == "train"] == [1, 2, 3, 4, 5, 6]
    assert sorted(os.listdir(f"{stage_path}/checkpoints")) == ["2.pt", "4.pt", "6.pt"]


def test_resume_continues_the_same_batch_order(config, ds_tokenized, tok, transformer):
    """Stopping and resuming sees the same minibatches as one uninterrupted run."""
    straight = dict(config, name="straight")
    stage = config["stages"][STAGE]
    straight["stages"] = {STAGE: dict(stage, train=dict(stage["train"], num_steps=4))}
    resumed = dict(config, name="resumed")
    resumed["stages"] = {STAGE: dict(stage, train=dict(stage["train"], num_steps=2))}

    def tokens_per_step(cfg):
        return [
            r["num_tokens"]
            for r in read_results(runs.get_stage_path(cfg, STAGE))
            if r["type"] == "train"
        ]

    torch.manual_seed(0)
    run_training(straight, transformer, ds_tokenized, tok)
    run_training(resumed, transformer, ds_tokenized, tok)
    resumed["stages"][STAGE]["train"]["num_steps"] = 4
    run_training(resumed, transformer, ds_tokenized, tok)

    assert tokens_per_step(resumed) == tokens_per_step(straight)


def test_final_step_is_checkpointed(config, transformer, ds_tokenized, tok):
    config["stages"][STAGE]["train"]["num_steps"] = 3
    stage_path = run_training(config, transformer, ds_tokenized, tok)
    assert sorted(os.listdir(f"{stage_path}/checkpoints")) == ["2.pt", "3.pt"]


def test_train_skips_when_already_complete(config, transformer, ds_tokenized, tok, monkeypatch):
    stage_path = run_training(config, transformer, ds_tokenized, tok)

    def fail(*args, **kwargs):
        raise AssertionError("the training loop should not run")

    monkeypatch.setattr(training, "supervised_loop", fail)
    run_training(config, transformer, ds_tokenized, tok)
    assert [r["step_no"] for r in read_results(stage_path) if r["type"] == "train"] == [1, 2, 3, 4]


# --- results recorded from the training code before training.train ran the loop itself ---

REFERENCE = json.loads((Path(__file__).parent / "fixtures/training_results.json").read_text())


def _flat(entry, prefix=""):
    """A results entry without its timings, nested values flattened, for pytest.approx."""
    flat = {}
    for key, value in entry.items():
        if isinstance(value, dict):
            flat.update(_flat(value, f"{prefix}{key}."))
        elif key != "tokens_per_sec":
            flat[prefix + key] = value
    return flat


def test_training_matches_the_recorded_results(config, raw_ds):
    """A short run, BLEU included. The tokenizer is BPE, whose training is deterministic, so
    the results repeat from one process to the next; WordPiece breaks ties randomly."""
    texts = list(raw_ds["train"]["text_en"]) + list(raw_ds["train"]["text_cy"])
    tok = train_tokenizer({"type": "bpe", "vocab_size": 300}, iter(texts))
    ds = tokenizer.tokenize_dataset(raw_ds, tok, config)
    config["tokenizer"]["vocab_size"] = len(tok)
    torch.manual_seed(0)
    model = build_model(config)

    stage_path = run_training(config, model, ds, tok)

    results = [_flat(entry) for entry in read_results(stage_path)]
    assert len(results) == len(REFERENCE)
    for entry, expected in zip(results, REFERENCE, strict=True):
        assert entry == pytest.approx(expected, rel=1e-5)
