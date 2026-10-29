import logging

from lab import runs, tokenizer
from lab.models import build_model

from src import data, dataloader, train
from src.config import Config


# Everything runs under a __main__ guard, because DataLoader workers re-import this module
def main() -> None:

    logging.basicConfig(level=logging.INFO)

    config = runs.parse_config(Config)
    runs.init_run(config, "pretrain", Config)

    # scripts/tokenizer.py trained the tokenizer; this only loads it
    token = tokenizer.load_tokenizer(runs.get_run_path(config))

    ds_raw = data.get_dataset_pretrain(config)

    ds = data.prepare_pretrain(ds_raw, token, config)

    dataloaders = dataloader.create_dataloaders_pretrain(ds, token, config)

    transformer = build_model(config)

    train.train("pretrain", transformer, dataloaders, token, config)


if __name__ == "__main__":
    main()
