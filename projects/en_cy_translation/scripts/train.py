import logging

from mlcore import runs
from mlcore.models import encoder_decoder

from src import dataloader, datasets, tokenizer, train


# Everything runs under a __main__ guard, because DataLoader workers re-import this module
def main() -> None:

    logging.basicConfig(level=logging.INFO)

    config = runs.parse_config()
    runs.init_run(config, train.STAGE)

    ds_raw = datasets.get_dataset(train.STAGE, config)

    token = tokenizer.get_tokenizer(ds_raw, config)

    ds_tokenized = tokenizer.tokenize_dataset(ds_raw, token, config)

    dataloaders = dataloader.create_dataloaders(ds_tokenized, config, train.STAGE)

    transformer = encoder_decoder.build_transformer(config)

    train.train(transformer, dataloaders, token, config)


if __name__ == "__main__":
    main()
