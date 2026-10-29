"""Every config in configs/ matches the schema, and agrees with the layout it replaced.

tests/fixtures/pre_registry.json holds each config as it was before named model components and
the stage-first layout, and the model each one built. Today's configs must build the same
model, and a run saved under the old layout must resume rather than be cleared.
"""

import glob
import hashlib
import json
import os

import pytest
import torch
import yaml
from mlcore import config as config_schema
from mlcore import runs
from mlcore.models import build_model

from src.config import Config

CONFIGS = sorted(glob.glob("configs/*.yaml"))
PRE_REGISTRY = json.load(open("tests/fixtures/pre_registry.json"))


def _load(path):
    return yaml.safe_load(open(path))


def test_every_config_has_a_pre_registry_record():
    assert [os.path.basename(p) for p in CONFIGS] == sorted(PRE_REGISTRY["configs"])


@pytest.mark.parametrize("path", CONFIGS)
def test_config_matches_the_schema(path):
    runs.load_config_file(path, Config)


@pytest.mark.parametrize("path", CONFIGS)
def test_config_builds_the_same_model_as_before(path):
    """Same parameter names, shapes and count as the code before named components."""
    with torch.device("meta"):
        model = build_model(_load(path))
    params = {k: list(v.shape) for k, v in model.state_dict().items()}
    layout = hashlib.sha256(json.dumps(sorted(params.items())).encode()).hexdigest()[:16]
    expected = PRE_REGISTRY["models"][os.path.basename(path)]
    assert sum(p.numel() for p in model.parameters()) == expected["count"]
    assert layout == expected["layout_sha"]


@pytest.mark.parametrize("path", CONFIGS)
def test_a_run_saved_under_the_old_layout_resumes(path):
    config = _load(path)
    old = PRE_REGISTRY["configs"][os.path.basename(path)]
    upgraded = runs.upgrade_legacy_config(old, config)
    for stage in config["stages"]:
        fingerprint = runs._stage_fingerprint(config, stage, Config)
        assert fingerprint is not None
        assert runs._stage_fingerprint(upgraded, stage, Config) == fingerprint, stage


@pytest.mark.parametrize("path", CONFIGS)
def test_a_changed_setting_in_an_old_run_is_still_noticed(path):
    config = _load(path)
    old = json.loads(json.dumps(PRE_REGISTRY["configs"][os.path.basename(path)]))
    stage = next(s for s in config["stages"] if config["stages"][s].get("train"))
    old["train"][stage]["learning_rate"] *= 2
    upgraded = runs.upgrade_legacy_config(old, config)
    assert runs._stage_fingerprint(upgraded, stage, Config) != runs._stage_fingerprint(
        config, stage, Config
    )


def test_a_misspelt_key_is_rejected_before_anything_runs(tmp_path):
    config = _load(CONFIGS[0])
    stage = next(s for s in config["stages"] if config["stages"][s].get("train"))
    train = config["stages"][stage]["train"]
    train["learning_rat"] = train.pop("learning_rate")
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(config_schema.ConfigError, match=f"stages\\.{stage}\\.train: unknown key"):
        runs.load_config_file(str(path), Config)
