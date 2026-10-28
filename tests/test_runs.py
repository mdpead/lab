"""Run and stage directories, and clearing a stage when the config it depends on changes."""

import copy
import os

import pytest

from mlcore import runs

STAGE = "s"


@pytest.fixture
def config(tmp_path):
    return {
        "name": "t",
        "seed": 0,
        "model": {"d_model": 32},
        "tokenizer": {"vocab_size": 100},
        "train": {STAGE: {"learning_rate": 1e-3, "num_steps": 4, "checkpoint_steps": 2}},
        "data": {STAGE: {"sample_size": 10}},
        "locations": {"models_dir": str(tmp_path)},
    }


def test_paths(config, tmp_path):
    assert runs.get_run_path(config) == f"{tmp_path}/t"
    assert runs.get_stage_path(config, STAGE) == f"{tmp_path}/t/{STAGE}"


def test_save_and_load_config_roundtrip(tmp_path, config):
    runs.save_config(tmp_path, config)
    assert runs.load_config(tmp_path) == config


def _init_with_marker(config):
    stage_path = runs.init_run(config, STAGE)
    marker = f"{stage_path}/marker"
    open(marker, "w").close()
    return stage_path, marker


def test_resumable_key_change_keeps_the_stage(config):
    stage_path, marker = _init_with_marker(config)

    config["train"][STAGE]["num_steps"] = 1000
    config["train"][STAGE]["checkpoint_steps"] = 50
    runs.init_run(config, STAGE)

    assert os.path.exists(marker)
    assert runs.load_config(stage_path)["train"][STAGE]["num_steps"] == 1000


@pytest.mark.parametrize(
    "section, key, value",
    [("train", "learning_rate", 5e-4), ("data", "sample_size", 20)],
)
def test_stage_setting_change_clears_the_stage(config, section, key, value):
    stage_path, marker = _init_with_marker(config)

    config[section][STAGE][key] = value
    runs.init_run(config, STAGE)

    assert not os.path.exists(marker)
    assert runs.load_config(stage_path)[section][STAGE][key] == value


def test_model_change_clears_the_stage(config):
    stage_path, marker = _init_with_marker(config)

    config["model"]["d_model"] = 64
    runs.init_run(config, STAGE)

    assert not os.path.exists(marker)


def test_another_stage_changing_keeps_this_one(config):
    _, marker = _init_with_marker(config)

    config["train"]["other"] = {"learning_rate": 1.0}
    runs.init_run(config, STAGE)

    assert os.path.exists(marker)


def test_fingerprint_does_not_mutate_config(config):
    original = copy.deepcopy(config)
    runs._stage_fingerprint(config, STAGE)
    assert config == original
