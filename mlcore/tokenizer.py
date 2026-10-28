import os
from collections.abc import Callable
from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from mlcore import runs

# Trains a tokenizer from a project's datasets and its config's `tokenizer` section
CreateTokenizer = Callable[[dict[str, Dataset], dict[str, Any]], PreTrainedTokenizerFast]


def save_tokenizer(tokenizer: PreTrainedTokenizerFast, dir: str) -> None:
    tokenizer.save_pretrained(f"{dir}/tokenizer")


def load_tokenizer(dir: str) -> PreTrainedTokenizerFast:
    if not os.path.isdir(f"{dir}/tokenizer"):
        raise FileNotFoundError(
            f"No tokenizer in {dir} — train one for this config first "
            "(the project's tokenizer step, or get_tokenizer)."
        )
    return PreTrainedTokenizerFast.from_pretrained(f"{dir}/tokenizer")


def get_tokenizer(
    ds: dict[str, Dataset], config: runs.Config, create_tokenizer: CreateTokenizer
) -> PreTrainedTokenizerFast:
    tokenizer_config = config["tokenizer"]
    run_path = runs.get_run_path(config)
    tokenizer_path = f"{run_path}/tokenizer"

    if os.path.isdir(tokenizer_path):
        return load_tokenizer(run_path)

    tokenizer = create_tokenizer(ds, tokenizer_config)
    save_tokenizer(tokenizer, run_path)

    return tokenizer
