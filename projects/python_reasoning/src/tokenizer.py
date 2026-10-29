import itertools
from typing import Any

from datasets import Dataset
from lab import runs
from lab import tokenizer as tokenizer_utils
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

from src import data


def create_tokenizer(
    ds: dict[str, Dataset], tokenizer_config: dict[str, Any]
) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # digit_split gives each digit its own token, so arithmetic is learnable.
    # Read from the config: changing it invalidates the saved tokenizer and checkpoints.
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
    if tokenizer_config.get("digit_split"):
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Digits(individual_digits=True), byte_level]
        )
    else:
        tokenizer.pre_tokenizer = byte_level
    tokenizer.decoder = decoders.ByteLevel()

    # Reasoning and answer delimiters, pipe-guarded so they can't collide with Python reprs.
    # Reserved with spare slots: the vocabulary is fixed once pretraining starts.
    special_tokens = [
        "<bos>",
        "<eos>",
        "<pad>",
        "<unk>",
        "<|think|>",
        "<|/think|>",
        "<|answer|>",
        "<|/answer|>",
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


def get_tokenizer(ds: dict[str, Dataset], config: runs.Config) -> PreTrainedTokenizerFast:
    return tokenizer_utils.get_tokenizer(ds, config, create_tokenizer)
