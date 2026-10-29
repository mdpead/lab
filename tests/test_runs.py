"""Run and stage directories, and clearing a stage when the config it depends on changes."""

import copy
import os
from dataclasses import dataclass
from typing import Any

import pytest
import yaml

from mlcore import runs
from mlcore.config import ConfigError, resumable

STAGE = "s"


@dataclass(frozen=True, kw_only=True)
class _Train:
    learning_rate: float
    num_steps: int = resumable()
    checkpoint_steps: int = resumable()
    label_smoothing: float = 0.0


@dataclass(frozen=True, kw_only=True)
class _Stage:
    train: _Train
    data: dict[str, Any] | None = None


@dataclass(frozen=True, kw_only=True)
class _Stages:
    s: _Stage | None = None
    other: _Stage | None = None


@dataclass(frozen=True, kw_only=True)
class Schema(runs.RunConfig):
    model: dict[str, Any]
    tokenizer: dict[str, Any]
    stages: _Stages


@pytest.fixture
def config(tmp_path):
    return {
        "name": "t",
        "seed": 0,
        "model": {"d_model": 32},
        "tokenizer": {"vocab_size": 100},
        "stages": {
            STAGE: {
                "train": {"learning_rate": 1e-3, "num_steps": 4, "checkpoint_steps": 2},
                "data": {"sample_size": 10},
            }
        },
        "locations": {"models_dir": str(tmp_path)},
    }


def test_paths(config, tmp_path):
    assert runs.get_run_path(config) == f"{tmp_path}/t"
    assert runs.get_stage_path(config, STAGE) == f"{tmp_path}/t/{STAGE}"


def test_save_and_load_config_roundtrip(tmp_path, config):
    runs.save_config(tmp_path, config)
    assert runs.load_config(tmp_path) == config


def _init_with_marker(config):
    stage_path = runs.init_run(config, STAGE, Schema)
    marker = f"{stage_path}/marker"
    open(marker, "w").close()
    return stage_path, marker


def test_resumable_key_change_keeps_the_stage(config):
    stage_path, marker = _init_with_marker(config)

    config["stages"][STAGE]["train"]["num_steps"] = 1000
    config["stages"][STAGE]["train"]["checkpoint_steps"] = 50
    runs.init_run(config, STAGE, Schema)

    assert os.path.exists(marker)
    assert runs.load_config(stage_path)["stages"][STAGE]["train"]["num_steps"] == 1000


@pytest.mark.parametrize(
    "section, key, value",
    [("train", "learning_rate", 5e-4), ("data", "sample_size", 20)],
)
def test_stage_setting_change_clears_the_stage(config, section, key, value):
    stage_path, marker = _init_with_marker(config)

    config["stages"][STAGE][section][key] = value
    runs.init_run(config, STAGE, Schema)

    assert not os.path.exists(marker)
    assert runs.load_config(stage_path)["stages"][STAGE][section][key] == value


def test_model_change_clears_the_stage(config):
    stage_path, marker = _init_with_marker(config)

    config["model"]["d_model"] = 64
    runs.init_run(config, STAGE, Schema)

    assert not os.path.exists(marker)


def test_another_stage_changing_keeps_this_one(config):
    _, marker = _init_with_marker(config)

    config["stages"]["other"] = {"train": {"learning_rate": 1.0}}
    runs.init_run(config, STAGE, Schema)

    assert os.path.exists(marker)


def test_fingerprint_does_not_mutate_config(config):
    original = copy.deepcopy(config)
    runs._stage_fingerprint(config, STAGE, Schema)
    assert config == original


def test_resumable_keys_come_from_the_schema(config):
    """A key marked resumable() in the schema changes without clearing; others clear."""
    stage_path, marker = _init_with_marker(config)

    config["stages"][STAGE]["train"]["checkpoint_steps"] = 7
    runs.init_run(config, STAGE, Schema)
    assert os.path.exists(marker)

    config["stages"][STAGE]["train"]["learning_rate"] = 9e-4
    runs.init_run(config, STAGE, Schema)
    assert not os.path.exists(marker)


def test_load_config_file_names_the_file_and_the_key(tmp_path, config):
    config["stages"][STAGE]["train"]["num_stpes"] = 3
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ConfigError, match=r"bad\.yaml: stages\.s\.train: unknown key 'num_stpes'"):
        runs.load_config_file(str(path), Schema)


def test_writing_out_a_default_does_not_clear_the_stage(config):
    """Fingerprints compare parsed configs, so an explicit default equals a missing one."""
    _, marker = _init_with_marker(config)

    config["stages"][STAGE]["train"]["label_smoothing"] = 0.0
    runs.init_run(config, STAGE, Schema)
    assert os.path.exists(marker)

    config["stages"][STAGE]["train"]["label_smoothing"] = 0.1
    runs.init_run(config, STAGE, Schema)
    assert not os.path.exists(marker)


def test_a_saved_config_that_no_longer_parses_clears_the_stage(config):
    stage_path, marker = _init_with_marker(config)

    saved = runs.load_config(stage_path)
    saved["stages"][STAGE]["train"]["learning_rate"] = "fast"
    runs.save_config(stage_path, saved)
    runs.init_run(config, STAGE, Schema)
    assert not os.path.exists(marker)


def test_upgrade_legacy_config_moves_stages_and_keeps_the_source_stage():
    current = {"stages": {"sft": {"train": {}, "init_from": {"stage": "pretrain"}}}}
    saved = {
        "name": "t",
        "model": {"type": "decoder_only"},
        "train": {"sft": {"learning_rate": 1e-3, "init_from_step": 200}, "pretrain": {}},
        "data": {"sft": {"tier": "easy"}},
    }
    upgraded = runs.upgrade_legacy_config(saved, current)
    assert "train" not in upgraded and "data" not in upgraded
    assert upgraded["stages"]["sft"] == {
        "train": {"learning_rate": 1e-3},
        "data": {"tier": "easy"},
        "init_from": {"stage": "pretrain", "step": 200},
    }
    assert upgraded["stages"]["pretrain"] == {"train": {}}
    assert saved["train"]["sft"]["init_from_step"] == 200, "the saved config is not modified"
