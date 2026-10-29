from collections.abc import Iterable

from datasets import Dataset
from lab import runs
from lab import tokenizer as tokenizer_utils
from transformers import PreTrainedTokenizerFast

from src import data


def corpus(ds: dict[str, Dataset]) -> Iterable[str]:
    """The tokenizer trains on the pretraining source files, with any markers removed."""
    return (data.strip_markers(example["content"]) for example in ds["train"])


def get_tokenizer(ds: dict[str, Dataset], config: runs.Config) -> PreTrainedTokenizerFast:
    return tokenizer_utils.get_tokenizer(ds, config, corpus)
