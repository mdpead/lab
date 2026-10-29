import itertools
from collections.abc import Iterable
from typing import Any

from lab import runs
from lab import tokenizer as tokenizer_utils
from transformers import PreTrainedTokenizerFast

from datasets import DatasetDict


def corpus(ds: DatasetDict) -> Iterable[str]:
    """The tokenizer trains on both languages' training sentences."""
    return itertools.chain(ds["train"]["text_en"], ds["train"]["text_cy"])


def _tokenize_text(
    text: dict[str, list[str]], tokenizer: PreTrainedTokenizerFast
) -> dict[str, list[list[int]]]:
    return {
        "text_en_tokenized": tokenizer(text["text_en"])["input_ids"],
        "text_cy_tokenized": tokenizer(text["text_cy"])["input_ids"],
    }


def tokenize_dataset(
    ds: DatasetDict, tokenizer: PreTrainedTokenizerFast, config: dict[str, Any]
) -> DatasetDict:
    ds = ds.map(lambda x: _tokenize_text(x, tokenizer), batched=True, batch_size=100000)
    ds = ds.map(
        lambda row: {
            "en_token_length": len(row["text_en_tokenized"]),
            "cy_token_length": len(row["text_cy_tokenized"]),
            # Source and target pad separately, so the longer side sets a row's cost
            "length": max(len(row["text_en_tokenized"]), len(row["text_cy_tokenized"])),
        },
    )
    max_length = config["model"]["max_length"]
    ds = ds.filter(
        lambda x: (x["en_token_length"] <= max_length) and (x["cy_token_length"] <= max_length)
    )
    return ds


def get_tokenizer(ds: DatasetDict, config: runs.Config) -> PreTrainedTokenizerFast:
    return tokenizer_utils.get_tokenizer(ds, config, corpus)
