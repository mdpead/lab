import json
import os

import pytest
import torch

from src import dataloader, train, utils


def lrs(warm_up_steps, num_steps, base_lr=1.0):
    optimiser = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=base_lr)
    scheduler = train.WarmupInverseSquareRootLR(optimiser, warm_up_steps)
    values = []
    for _ in range(num_steps):
        values.append(scheduler.get_last_lr()[0])
        optimiser.step()
        scheduler.step()
    return values


def test_lr_schedule_warms_up_linearly_then_decays():
    values = lrs(warm_up_steps=4, num_steps=20)
    assert values[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert values[4] == pytest.approx((4 / 5) ** 0.5)
    assert values[15] == pytest.approx((4 / 16) ** 0.5)
    assert max(values) == pytest.approx(1.0)
    assert all(a > b for a, b in zip(values[3:], values[4:]))


def test_lr_schedule_scales_with_base_lr():
    assert lrs(4, 10, base_lr=1e-3) == pytest.approx([v * 1e-3 for v in lrs(4, 10)])


def test_create_training_objects(transformer, config):
    criterion, optimiser, scheduler, scaler = train.create_training_objects(transformer, config)
    assert criterion.ignore_index == config["tokenizer"]["pad_token_id"]
    assert criterion.label_smoothing == config["train"]["label_smoothing"]
    assert list(optimiser.param_groups[0]["betas"]) == config["train"]["adam_betas"]
    assert isinstance(scheduler, train.WarmupInverseSquareRootLR)


def test_checkpoint_roundtrip(tmp_path, transformer, config):
    run = train.create_run(transformer, config)
    train.save_checkpoint(
        transformer,
        run["optimiser"],
        run["lr_scheduler"],
        run["scaler"],
        tmp_path,
        7,
        [{"step_no": 7}],
    )

    checkpoint = train.load_checkpoint(tmp_path, 7, torch.device("cpu"))

    assert checkpoint["step_no"] == 7
    for key, value in transformer.state_dict().items():
        assert torch.equal(checkpoint["model_state_dict"][key], value)
    assert json.load(open(tmp_path / "results.json")) == [{"step_no": 7}]


def test_save_checkpoint_unwraps_compiled_model(tmp_path, transformer, config):
    run = train.create_run(transformer, config)
    wrapper = torch.nn.Module()
    wrapper._orig_mod = transformer
    train.save_checkpoint(
        wrapper, run["optimiser"], run["lr_scheduler"], run["scaler"], tmp_path, 1, []
    )

    checkpoint = train.load_checkpoint(tmp_path, 1)

    assert checkpoint["model_state_dict"].keys() == transformer.state_dict().keys()


def test_load_run_without_checkpoints_raises(tmp_path, transformer, config):
    os.makedirs(tmp_path / "checkpoints")
    with pytest.raises(FileNotFoundError):
        train.load_run(str(tmp_path), transformer, config)


def test_get_run_creates_new_run(transformer, config):
    run = train.get_run(config, transformer)
    assert run["step_no"] == 0
    assert run["results"] == []
    assert run["run_path"] == utils.get_run_path(config)


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
    utils.init_run(config)
    loaders = dataloader.create_dataloaders(ds_tokenized, config)
    return train.train(transformer, loaders, tok, config)


def test_train_end_to_end(config, transformer, ds_tokenized, tok):
    run_training(config, transformer, ds_tokenized, tok)
    run_path = utils.get_run_path(config)

    assert sorted(os.listdir(f"{run_path}/checkpoints")) == ["2.pt", "4.pt"]
    results = json.load(open(f"{run_path}/results.json"))
    train_results = [r for r in results if r["type"] == "train"]
    validation_results = [r for r in results if r["type"] == "validation"]
    assert [r["step_no"] for r in train_results] == [1, 2, 3, 4]
    assert [r["step_no"] for r in validation_results] == [2, 4]
    assert all(r["loss"] > 0 and r["num_tokens"] > 0 for r in train_results)
    assert all("bleu" in r["validation_metrics"] for r in validation_results)
    assert train_results[0]["learning_rate"] == pytest.approx(config["train"]["learning_rate"])


def test_train_resumes_from_checkpoint(config, transformer, ds_tokenized, tok):
    run_training(config, transformer, ds_tokenized, tok)
    run_path = utils.get_run_path(config)

    config["train"]["num_steps"] = 6
    run = train.get_run(config, transformer)
    assert run["step_no"] == 4
    assert len(run["results"]) == 6

    run_training(config, transformer, ds_tokenized, tok)
    results = json.load(open(f"{run_path}/results.json"))
    assert [r["step_no"] for r in results if r["type"] == "train"] == [1, 2, 3, 4, 5, 6]
    assert sorted(os.listdir(f"{run_path}/checkpoints")) == ["2.pt", "4.pt", "6.pt"]


def test_train_skips_when_already_complete(config, transformer, ds_tokenized, tok, monkeypatch):
    run_training(config, transformer, ds_tokenized, tok)

    def fail(*args, **kwargs):
        raise AssertionError("train_loop should not run")

    monkeypatch.setattr(train, "train_loop", fail)
    results = run_training(config, transformer, ds_tokenized, tok)
    assert [r["step_no"] for r in results if r["type"] == "train"] == [1, 2, 3, 4]
