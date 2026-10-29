"""One-off: rewrite the config.json files saved in run directories to the current config layout.

Runs started before models were built from named components saved their config with flat
model sizes and separate `train.<stage>` and `data.<stage>` sections. Runs started before
tokenizers moved into the lab package saved a tokenizer section without a `type`. The code no
longer reads either, so a saved config in them no longer matches and the stage, or the saved
tokenizer, would be cleared. This rewrites each one to the layout configs/*.yaml now use,
keeping the tokenizer and checkpoints, so rerunning a stage resumes it. Each file it changes
is first copied to config.json.bak, unless an earlier run of this script already made one.

    python tools/migrate_saved_configs.py projects/python_reasoning/models --dry-run
    python tools/migrate_saved_configs.py projects/python_reasoning/models \
        projects/en_cy_translation/models

Delete this script once every run directory has been migrated.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

# The earlier stage each stage started from. The old scripts fixed this: sft started from
# pretrain only when `init_from_step` was set, and rl always started from sft.
SOURCE_STAGE = {"sft": "pretrain", "rl": "sft"}
ALWAYS_SEEDED = {"rl"}


def migrate_model(model: dict[str, Any]) -> dict[str, Any]:
    """A flat model section, as either family's old configs listed it, in the current form."""
    if "num_enc_layers" in model:
        return {
            "type": "encoder_decoder",
            "d_model": model["d_model"],
            "num_enc_layers": model["num_enc_layers"],
            "num_dec_layers": model["num_dec_layers"],
            "max_length": model["max_length"],
            "dropout": model["dropout"],
            "norm": {"type": "layer"},
            "position": {"type": "sinusoidal"},
            "attention": {"type": "mha", "num_heads": model["num_heads"]},
            "feedforward": {"type": "relu", "d_ff": model["d_ff"], "bias": True},
        }
    return {
        "type": "decoder_only",
        "d_model": model["d_model"],
        "num_layers": model["num_layers"],
        "max_length": model["max_length"],
        "dropout": model["dropout"],
        "norm": {"type": "rms"},
        "position": {"type": "rope"},
        "attention": {
            "type": "gqa",
            "num_heads": model["num_heads"],
            "num_kv_heads": model["num_kv_heads"],
            "d_h": model["d_h"],
        },
        "feedforward": {
            "type": "moe",
            "d_ff": model["d_ff"],
            "num_experts": model["num_experts"],
            "top_k": model["top_k"],
            "capacity_factor": model["capacity_factor"],
        },
    }


# python_reasoning's markers and spare slots, which its tokenizer code listed before
REASONING_SPECIAL_TOKENS = ["<|think|>", "<|/think|>", "<|answer|>", "<|/answer|>"]


def migrate_tokenizer(tokenizer: dict[str, Any], model_type: str) -> dict[str, Any]:
    """A tokenizer section without a type, as the project owning `model_type` built it."""
    if model_type == "encoder_decoder":
        # en_cy_translation's WordPiece; the pad id now comes from the tokenizer itself
        settings = {k: v for k, v in tokenizer.items() if k != "pad_token_id"}
        return {"type": "wordpiece", **settings, "strip_accents": True}
    return {
        "type": "bpe",
        **tokenizer,
        "special_tokens": REASONING_SPECIAL_TOKENS,
        "reserved_tokens": 16,
    }


def migrate_layout(config: dict[str, Any]) -> dict[str, Any]:
    """Named model components and a stages section, from the flat layout."""
    train, data = config.get("train", {}), config.get("data", {})
    stages: dict[str, Any] = {}
    for stage in {**train, **data}:
        stage_config: dict[str, Any] = {}
        train_config = dict(train.get(stage) or {})
        step = train_config.pop("init_from_step", None)
        if stage in SOURCE_STAGE and (step is not None or stage in ALWAYS_SEEDED):
            source = {"stage": SOURCE_STAGE[stage]}
            if step is not None:
                source["step"] = step
            stage_config["init_from"] = source
        if stage in train:
            stage_config["train"] = train_config
        if stage in data:
            stage_config["data"] = data[stage]
        stages[stage] = stage_config

    migrated = {k: v for k, v in config.items() if k not in ("train", "data")}
    migrated["model"] = migrate_model(config["model"])
    migrated["stages"] = stages
    return migrated


def is_old_layout(config: dict[str, Any]) -> bool:
    return "stages" not in config and ("train" in config or "type" not in config.get("model", {}))


def migrate(config: dict[str, Any]) -> dict[str, Any]:
    """A saved config in the current layout. Values are kept; only their places change."""
    if is_old_layout(config):
        config = migrate_layout(config)
    if "tokenizer" in config and "type" not in config["tokenizer"]:
        tokenizer = migrate_tokenizer(config["tokenizer"], config["model"]["type"])
        config = {**config, "tokenizer": tokenizer}
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models_dirs", nargs="+", type=Path, help="Directories of runs")
    parser.add_argument("--dry-run", action="store_true", help="Report, change nothing")
    args = parser.parse_args()

    changed = 0
    for models_dir in args.models_dirs:
        if not models_dir.is_dir():
            print(f"skipping {models_dir}: not a directory", file=sys.stderr)
            continue
        for path in sorted(models_dir.rglob("config.json")):
            config = json.loads(path.read_text())
            migrated = migrate(config)
            if migrated == config:
                print(f"already current  {path}")
                continue
            print(f"{'would migrate' if args.dry_run else 'migrating'}    {path}")
            if not args.dry_run:
                backup = path.with_name("config.json.bak")
                if not backup.exists():
                    shutil.copy2(path, backup)
                path.write_text(json.dumps(migrated))
            changed += 1
    print(f"{changed} config(s) {'to migrate' if args.dry_run else 'migrated'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
