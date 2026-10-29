import os

# Set before torch loads, as in the training scripts
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import logging
from typing import Any

import torch
from lab import generation, runs, tokenizer, training
from lab.models import build_model

from src import data, rl
from src.config import Config


# Greedy pass@1 on a test set larger than RL validation's ~400 prompts.
# Functions come from a seed no stage trains on; only those on every stage's test side are kept.
def eval_rows(config: runs.Config, num_records: int, seed: int) -> list[dict[str, Any]]:
    ds_config = {**config["stages"]["rl"]["data"], "num_records": num_records}
    ratio = min(config["stages"]["sft"]["data"]["test_split_ratio"], ds_config["test_split_ratio"])
    rows: list[dict[str, Any]] = []
    for record in data._synth_records(ds_config, seed, with_trace=False):
        if not data._is_test(record["source"], ratio):
            continue
        for task in ds_config["tasks"]:
            prompt, _ = data._TASK_FORMATTERS[task](record, False)
            rows.append(
                {
                    "prompt": prompt,
                    "task": task,
                    "source": record["source"],
                    "args": record["args"],
                    "result": record["result"],
                    "n_executed": record["n_executed"],
                }
            )
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config name (e.g. base_lr3)")
    parser.add_argument("--stage", default="sft", help="Stage whose checkpoint to score")
    parser.add_argument("--step", type=int, required=True, help="Checkpoint step")
    parser.add_argument("--num-records", type=int, default=150000, help="Before the split")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    config = runs.load_config_file(f"configs/{args.config}.yaml", Config)

    # sft draws from seed, rl from seed + 1
    rows = eval_rows(config, args.num_records, config["seed"] + 2)
    num_functions = len({row["source"] for row in rows})
    logging.info(f"eval: {len(rows)} prompts over {num_functions} functions")

    device = torch.device(config["stages"][args.stage]["train"]["device"])
    transformer = build_model(config)
    training.load_model_weights(
        transformer, runs.get_stage_path(config, args.stage), device, step=args.step
    )
    tok = tokenizer.load_tokenizer(runs.get_run_path(config))

    max_new_tokens = config["stages"]["rl"]["train"]["max_new_tokens"]
    passed: list[bool] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        texts, _ = generation.generate_texts(
            transformer, tok, [row["prompt"] for row in batch], device, max_new_tokens
        )
        passed += [bool(s) for s in rl.score(batch, texts)]
        logging.info(f"eval: {len(passed)}/{len(rows)}, pass rate {sum(passed) / len(passed):.3f}")

    result = {
        "config": args.config,
        "stage": args.stage,
        "step": args.step,
        "pass_rate": sum(passed) / len(passed),
        "num_prompts": len(rows),
        "num_functions": num_functions,
        # Per prompt, in a fixed order, so runs can be compared prompt by prompt
        "passed": passed,
        "sources": [data._hash_position(row["source"]) for row in rows],
    }
    out_dir = f"{runs.get_run_path(config)}/eval"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{args.stage}_{args.step}.json"
    with open(out_path, "w") as f:
        json.dump(result, f)
    logging.info(f"eval: pass rate {result['pass_rate']:.4f}, written to {out_path}")


if __name__ == "__main__":
    main()
