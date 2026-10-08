from src import data, model, tokenizer, train, utils, dataloader
import logging


# Supervised fine-tuning on synth tasks: prompt in, trace and answer out, loss on the completion
def main():

    logging.basicConfig(level=logging.INFO)

    config = utils.parse_config()
    utils.init_run(config, "sft")

    # Pretraining trained the tokenizer; this only loads it
    token = tokenizer.load_tokenizer(utils.get_run_path(config))

    ds_raw = data.get_dataset_sft(config)

    ds = data.prepare_sft(ds_raw, token, config)

    logging.info(f"sft rows: {  {split: len(rows) for split, rows in ds.items()} }")

    transformer = model.build_transformer(config)

    # Rows are ragged, so batches come from a TokenSampler against a token budget
    dataloaders = dataloader.create_dataloaders_sft(ds, token, config)

    # Starts from the pretrained weights only; train.sft.init_from_step pins the checkpoint
    train.train(
        "sft", transformer, dataloaders, token, config,
        init_from=utils.get_stage_path(config, "pretrain"),
    )


if __name__ == "__main__":
    main()
