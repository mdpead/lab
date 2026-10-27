import logging

from mlcore import runs

from src import data, tokenizer


# Train the BPE tokenizer on the pretrain corpus. Later stages load it from the run
# directory, so this runs once per config.
def main() -> None:

    logging.basicConfig(level=logging.INFO)

    config = runs.parse_config()
    run_path = runs.init_run_dir(config)

    ds_raw = data.get_dataset_pretrain(config)

    token = tokenizer.get_tokenizer(ds_raw, config)

    logging.info(f"tokenizer: {len(token)} tokens in {run_path}/tokenizer")


if __name__ == "__main__":
    main()
