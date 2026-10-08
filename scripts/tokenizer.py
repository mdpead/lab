from src import data, tokenizer, utils
import logging


# Train the BPE tokenizer on the pretrain corpus. Every later stage loads it from the run
# directory, so this runs once per config, before any of them.
def main():

    logging.basicConfig(level=logging.INFO)

    config = utils.parse_config()
    run_path = utils.init_run_dir(config)

    ds_raw = data.get_dataset_pretrain(config)

    token = tokenizer.get_tokenizer(ds_raw, config)

    logging.info(f"tokenizer: {len(token)} tokens in {run_path}/tokenizer")


if __name__ == "__main__":
    main()
