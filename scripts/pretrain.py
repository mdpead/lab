from src import data, tokenizer, model, train, utils, dataloader
import logging


# Everything runs under a __main__ guard, because DataLoader workers re-import this module
def main():

    logging.basicConfig(level=logging.INFO)

    config = utils.parse_config()
    utils.init_run(config, "pretrain")

    ds_raw = data.get_dataset_pretrain(config)

    # Trains the BPE on first run and loads it after, so the raw dataset must exist first
    token = tokenizer.get_tokenizer(ds_raw, config)

    ds = data.prepare_pretrain(ds_raw, token, config)

    dataloaders = dataloader.create_dataloaders_pretrain(ds, token, config)

    transformer = model.build_transformer(config)

    train.train("pretrain", transformer, dataloaders, token, config)


if __name__ == "__main__":
    main()
