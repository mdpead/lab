import os

# Set before torch loads. Varying batch shapes fragment the allocator until the card overflows.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging

from lab import runs, tokenizer
from lab.models import build_model

from src import data, dataloader, rl
from src.config import Config


# RL on synth tasks: sample rollouts, score by executing them, update on GRPO advantages.
# See src/rl.py.
def main() -> None:

    logging.basicConfig(level=logging.INFO)

    config = runs.parse_config(Config)
    runs.init_run(config, "rl", Config)

    token = tokenizer.load_tokenizer(runs.get_run_path(config))

    ds_raw = data.get_dataset_rl(config)

    ds = data.prepare_rl(ds_raw, token, config)

    logging.info(f"rl prompts: { {split: len(rows) for split, rows in ds.items()} }")

    dataloaders = dataloader.create_dataloaders_rl(ds, token, config)

    transformer = build_model(config)

    # Starts from SFT (stages.rl.init_from): a pretrained model never emits the answer
    # marker, so every reward would be 0.
    rl.train_rl("rl", transformer, dataloaders, token, config)


if __name__ == "__main__":
    main()
