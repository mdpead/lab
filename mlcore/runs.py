import argparse
import json
import logging
import os
import random
import shutil
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml

from mlcore import config as config_schema
from mlcore.config import Config
from mlcore.models import upgrade_legacy_model

__all__ = ["Config", "Locations", "RunConfig"]


@dataclass(frozen=True, kw_only=True)
class Locations:
    models_dir: str
    data_dir: str | None = None


@dataclass(frozen=True, kw_only=True)
class RunConfig:
    """The top-level settings mlcore reads. A project's schema subclasses it, adding its
    `model`, `tokenizer` and `stages` sections."""

    name: str
    seed: int
    locations: Locations


def get_run_path(config: Config) -> str:
    """Run-level artifacts shared by every stage: the tokenizer."""
    return f"{config['locations']['models_dir']}/{config['name']}"


def get_stage_path(config: Config, stage: str) -> str:
    """Per-stage artifacts: checkpoints, results, and that stage's config snapshot."""
    return f"{get_run_path(config)}/{stage}"


def load_config_file(path: str, schema: type) -> Config:
    """Read a YAML config and check it against `schema`, failing before anything runs.

    Returns the parsed dict, which is what the rest of the code reads.
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    config_schema.parse(schema, config, path)
    return config


def parse_config(schema: type) -> Config:
    """--config <name> -> configs/<name>.yaml, checked against `schema`. Shared by the stage
    scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config name (e.g. base, test)")
    args = parser.parse_args()
    return load_config_file(f"configs/{args.config}.yaml", schema)


# Sections every stage's checkpoints depend on, whichever stage is running.
SHARED_SECTIONS = ("name", "seed", "model", "tokenizer")


def _stage_fingerprint(config: Config, stage: str, schema: type) -> Any:
    """The part of a config that this stage's checkpoints depend on, or None if it no longer
    matches `schema`.

    Scoped to one stage, and compared after parsing, so a default written out explicitly is
    the same as one left out. Fields marked `resumable()` are left out: they can change
    mid-run.
    """
    stages_schema = config_schema.field_type(schema, "stages")
    sections = [
        (key, config_schema.field_hint(schema, key), config.get(key)) for key in SHARED_SECTIONS
    ]
    sections.append(
        ("stage", config_schema.field_hint(stages_schema, stage), config["stages"].get(stage))
    )
    try:
        return {
            key: config_schema.canonical(config_schema.parse(hint, value, path=key))
            for key, hint, value in sections
        }
    except config_schema.ConfigError as e:
        logging.info(f"Saved config no longer matches the schema: {e}")
        return None


def upgrade_legacy_config(saved: Config, current: Config) -> Config:
    """A saved config in today's layout, from either layout it may have been saved in.

    Configs saved before the stage-first layout kept each stage's settings under `train` and
    `data`, recorded only the step a stage started from (`train.<stage>.init_from_step`), and
    listed flat model sizes. The stage it started from was fixed by the project's scripts, so
    it is taken from the current config.
    """
    upgraded = dict(saved)
    if isinstance(saved.get("model"), dict):
        upgraded["model"] = upgrade_legacy_model(saved["model"])
    if "stages" in saved:
        return upgraded

    train, data = saved.get("train", {}), saved.get("data", {})
    stages = {}
    for stage in {**train, **data}:
        stage_config: dict[str, Any] = {}
        train_config = dict(train.get(stage) or {})
        step = train_config.pop("init_from_step", None)
        if stage in train:
            stage_config["train"] = train_config
        if stage in data:
            stage_config["data"] = data[stage]
        source = (current.get("stages", {}).get(stage) or {}).get("init_from")
        if source is not None or step is not None:
            stage_config["init_from"] = {"stage": (source or {}).get("stage"), "step": step}
        stages[stage] = stage_config
    upgraded.pop("train", None)
    upgraded.pop("data", None)
    return {**upgraded, "stages": stages}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _start_file_log(stage_path: str, stage: str, cleared: bool) -> str:
    """Tee the root logger into the stage directory, with timestamps, in append mode."""
    log_path = f"{stage_path}/{stage}.log"
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.info(f"--- {stage} starting, logging to {log_path} ---")
    if cleared:
        # The warning that triggered the clear was logged before this handler existed
        logging.info(f"{stage_path} was cleared first: the saved config no longer matched")
    return log_path


def init_run_dir(config: Config) -> str:
    """Prepare the run directory, clearing the tokenizer if its config moved."""
    set_seed(config["seed"])
    run_path = get_run_path(config)
    os.makedirs(run_path, exist_ok=True)

    # Tokenizer loads from disk whenever it exists, so a changed config needs its own check.
    # Otherwise it would be silently ignored.
    if os.path.exists(f"{run_path}/config.json"):
        existing = load_config(run_path)
        if existing.get("tokenizer") != config.get("tokenizer"):
            logging.warning("Tokenizer config differs from saved config — clearing tokenizer.")
            shutil.rmtree(f"{run_path}/tokenizer", ignore_errors=True)
    save_config(run_path, config)
    return run_path


def init_run(config: Config, stage: str, schema: type) -> str:
    """Prepare this stage's directory, clearing it if the config it depends on moved.

    Settings marked `resumable()` in `schema` can change without clearing it. A config saved
    in an older layout is upgraded before the comparison, so its run resumes.
    """
    init_run_dir(config)
    stage_path = get_stage_path(config, stage)
    os.makedirs(stage_path, exist_ok=True)

    cleared = False
    if os.path.exists(f"{stage_path}/config.json"):
        existing = upgrade_legacy_config(load_config(stage_path), config)
        saved_fingerprint = _stage_fingerprint(existing, stage, schema)
        if saved_fingerprint is None or saved_fingerprint != _stage_fingerprint(
            config, stage, schema
        ):
            logging.warning(f"{stage} config differs from saved config — clearing {stage_path}.")
            shutil.rmtree(stage_path)
            os.makedirs(stage_path)
            cleared = True
    save_config(stage_path, config)

    # Attached only now, after any clear of the directory the log file lives in
    _start_file_log(stage_path, stage, cleared)
    return stage_path


def save_config(run_path: str, config: Config) -> str:
    filepath = f"{run_path}/config.json"
    json.dump(config, open(filepath, "w"))
    return filepath


def load_config(run_path: str) -> Config:
    filepath = f"{run_path}/config.json"
    with open(filepath) as f:
        config = json.load(f)
    return config
