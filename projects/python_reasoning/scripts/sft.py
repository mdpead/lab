import os

# Set before torch loads. Varying batch shapes fragment the allocator until the card overflows.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging

from mlcore import runs, tokenizer
from mlcore.models import decoder_only

from src import data, dataloader, train


# Supervised fine-tuning on synth tasks: prompt in, trace and answer out, loss on the completion
def main() -> None:

    logging.basicConfig(level=logging.INFO)

    config = runs.parse_config()
    runs.init_run(config, "sft")

    # scripts/tokenizer.py trained the tokenizer; this only loads it
    token = tokenizer.load_tokenizer(runs.get_run_path(config))

    ds_raw = data.get_dataset_sft(config)

    ds = data.prepare_sft(ds_raw, token, config)

    logging.info(f"sft rows: { {split: len(rows) for split, rows in ds.items()} }")

    transformer = decoder_only.build_transformer(config)

    # Rows are ragged, so batches come from a TokenSampler against a token budget
    dataloaders = dataloader.create_dataloaders_sft(ds, token, config)

    # train.sft.init_from_step pins the pretrain checkpoint that seeds this stage.
    # Without it, sft trains from random init.
    init_from = None
    if "init_from_step" in config["train"]["sft"]:
        init_from = runs.get_stage_path(config, "pretrain")
    else:
        logging.info("sft: no init_from_step, training from random init")
    train.train("sft", transformer, dataloaders, token, config, init_from=init_from)


if __name__ == "__main__":
    main()
