"""Every config in configs/ matches the schema and builds the same model as before.

tests/fixtures/model_layouts.json records the model each config built before models were
assembled from named components: its parameter count and a hash of its names and shapes.
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
MODEL_LAYOUTS = json.load(open("tests/fixtures/model_layouts.json"))["models"]


def _load(path):
    return yaml.safe_load(open(path))


def test_every_config_has_a_recorded_layout():
    assert [os.path.basename(p) for p in CONFIGS] == sorted(MODEL_LAYOUTS)


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
    expected = MODEL_LAYOUTS[os.path.basename(path)]
    assert sum(p.numel() for p in model.parameters()) == expected["count"]
    assert layout == expected["layout_sha"]


def test_a_misspelt_key_is_rejected_before_anything_runs(tmp_path):
    config = _load(CONFIGS[0])
    stage = next(s for s in config["stages"] if config["stages"][s].get("train"))
    train = config["stages"][stage]["train"]
    train["learning_rat"] = train.pop("learning_rate")
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(config_schema.ConfigError, match=f"stages\\.{stage}\\.train: unknown key"):
        runs.load_config_file(str(path), Config)


def test_the_test_suite_config_matches_the_schema():
    """conftest's config is what most tests run on, so it must be a valid config too."""
    from tests.conftest import BASE_CONFIG

    config_schema.parse(Config, BASE_CONFIG)
