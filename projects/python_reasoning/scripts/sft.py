import os

# Set before torch loads. Varying batch shapes fragment the allocator until the card overflows.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging

from mlcore import runs, tokenizer
from mlcore.models import build_model

from src import data, dataloader, train
from src.config import Config


# Supervised fine-tuning on synth tasks: prompt in, trace and answer out, loss on the completion
def main() -> None:

    logging.basicConfig(level=logging.INFO)

    config = runs.parse_config(Config)
    runs.init_run(config, "sft", Config)

    # scripts/tokenizer.py trained the tokenizer; this only loads it
    token = tokenizer.load_tokenizer(runs.get_run_path(config))

    ds_raw = data.get_dataset_sft(config)

    ds = data.prepare_sft(ds_raw, token, config)

    logging.info(f"sft rows: { {split: len(rows) for split, rows in ds.items()} }")

    transformer = build_model(config)

    # Rows are ragged, so batches come from a TokenSampler against a token budget
    dataloaders = dataloader.create_dataloaders_sft(ds, token, config)

    # stages.sft.init_from names the pretrain checkpoint that seeds this stage.
    # Without it, sft trains from random init.
    if config["stages"]["sft"].get("init_from") is None:
        logging.info("sft: no init_from, training from random init")
    train.train("sft", transformer, dataloaders, token, config)


if __name__ == "__main__":
    main()
