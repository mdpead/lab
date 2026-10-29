"""Every config in configs/ matches the schema, and fingerprints match the old hard-coded rule."""

import glob

import pytest
import yaml
from mlcore import config as config_schema
from mlcore import runs

from src.config import Config

CONFIGS = sorted(glob.glob("configs/*.yaml"))

# The fingerprint rule before the schema existed, kept to show the schema reproduces it, so
# runs saved before the change resume rather than being cleared
OLD_RESUMABLE_KEYS = {
    "num_steps",
    "checkpoint_steps",
    "keep_checkpoints",
    "validation_steps",
    "validation_batches",
    "minibatch_token_size",
    "cache_clear_steps",
    "num_workers",
    "prefetch_factor",
    "compile_model",
}


def _old_fingerprint(config, stage):
    fingerprint = {key: config.get(key) for key in runs.SHARED_SECTIONS}
    train_config = config.get("train", {}).get(stage) or {}
    fingerprint["train"] = {k: v for k, v in train_config.items() if k not in OLD_RESUMABLE_KEYS}
    fingerprint["data"] = config.get("data", {}).get(stage)
    return fingerprint


def test_there_are_configs_to_check():
    assert CONFIGS


@pytest.mark.parametrize("path", CONFIGS)
def test_config_matches_the_schema(path):
    runs.load_config_file(path, Config)


@pytest.mark.parametrize("path", CONFIGS)
def test_fingerprint_is_unchanged_for_every_stage(path):
    config = yaml.safe_load(open(path))
    for stage in config["train"]:
        resumable = config_schema.resumable_keys(config_schema.stage_class(Config, stage))
        assert runs._stage_fingerprint(config, stage, resumable) == _old_fingerprint(config, stage)


def test_a_misspelt_key_is_rejected_before_anything_runs(tmp_path):
    config = yaml.safe_load(open(CONFIGS[0]))
    stage = next(iter(config["train"]))
    config["train"][stage]["learning_rat"] = config["train"][stage].pop("learning_rate")
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(config_schema.ConfigError, match=f"train\\.{stage}: unknown key"):
        runs.load_config_file(str(path), Config)
