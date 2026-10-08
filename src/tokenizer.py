import os
import itertools
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
from transformers import PreTrainedTokenizerFast
from src import data, utils


def create_tokenizer(ds, tokenizer_config):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # digit_split gives every digit its own token, so the model can learn arithmetic on
    # digits rather than memorise each number as an unrelated id. Read from the config so
    # that changing it invalidates the saved tokenizer and checkpoints.
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    if tokenizer_config.get("digit_split"):
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Digits(individual_digits=True), byte_level]
        )
    else:
        tokenizer.pre_tokenizer = byte_level
    tokenizer.decoder = decoders.ByteLevel()

    # Reasoning and answer delimiters, pipe-guarded so they cannot collide with Python
    # reprs like "<class 'int'>". Reserved now, with spare slots, because the vocabulary
    # is fixed once pretraining starts.
    special_tokens = [
        "<bos>", "<eos>", "<pad>", "<unk>",
        "<|think|>", "<|/think|>",
        "<|answer|>", "<|/answer|>",
        *[f"<|reserved_{i}|>" for i in range(16)],
    ]
    trainer = trainers.BpeTrainer(
        vocab_size=tokenizer_config["vocab_size"],
        special_tokens=special_tokens,
    )

    train_iter = itertools.islice(
        (data.strip_markers(example["content"]) for example in ds["train"]),
        tokenizer_config["training_size"],
    )
    tokenizer.train_from_iterator(train_iter, trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )


def save_tokenizer(tokenizer, dir):
    tokenizer.save_pretrained(f"{dir}/tokenizer")


def load_tokenizer(dir):
    if not os.path.isdir(f"{dir}/tokenizer"):
        raise FileNotFoundError(
            f"No tokenizer in {dir} — run `python -m scripts.tokenizer` for this config first."
        )
    return PreTrainedTokenizerFast.from_pretrained(f"{dir}/tokenizer")


def get_tokenizer(ds, config):
    tokenizer_config = config["tokenizer"]
    run_path = utils.get_run_path(config)
    tokenizer_path = f"{run_path}/tokenizer"

    if os.path.isdir(tokenizer_path):
        return load_tokenizer(run_path)

    tokenizer = create_tokenizer(ds, tokenizer_config)
    save_tokenizer(tokenizer, run_path)

    return tokenizer
