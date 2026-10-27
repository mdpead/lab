import logging

from mlcore import runs, tokenizer
from mlcore.models import decoder_only

from src import data, dataloader, train


# Everything runs under a __main__ guard, because DataLoader workers re-import this module
def main() -> None:

    logging.basicConfig(level=logging.INFO)

    config = runs.parse_config()
    runs.init_run(config, "pretrain")

    # scripts/tokenizer.py trained the tokenizer; this only loads it
    token = tokenizer.load_tokenizer(runs.get_run_path(config))

    ds_raw = data.get_dataset_pretrain(config)

    ds = data.prepare_pretrain(ds_raw, token, config)

    dataloaders = dataloader.create_dataloaders_pretrain(ds, token, config)

    transformer = decoder_only.build_transformer(config)

    train.train("pretrain", transformer, dataloaders, token, config)


if __name__ == "__main__":
    main()
