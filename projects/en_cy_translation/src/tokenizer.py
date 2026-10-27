import itertools
import os
from typing import Any

from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

from datasets import DatasetDict
from src import utils


def create_tokenizer(ds: DatasetDict, tokenizer_config: dict[str, Any]) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    special_tokens = ["[BOS]", "[EOS]", "[PAD]", "[MASK]", "[UNK]"]
    tokenizer.model = models.WordPiece(unk_token="[UNK]")
    trainer = trainers.WordPieceTrainer(
        vocab_size=tokenizer_config["vocab_size"], special_tokens=special_tokens
    )
    combined_text = itertools.chain(ds["train"]["text_en"], ds["train"]["text_cy"])
    train_iter = itertools.islice(combined_text, tokenizer_config["training_size"])
    tokenizer.train_from_iterator(train_iter, trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
    )


def save_tokenizer(tokenizer: PreTrainedTokenizerFast, dir: str) -> None:
    tokenizer.save_pretrained(f"{dir}/tokenizer")


def load_tokenizer(dir: str) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast.from_pretrained(f"{dir}/tokenizer")


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
        },
    )
    max_length = config["model"]["max_length"]
    ds = ds.filter(
        lambda x: (x["en_token_length"] <= max_length) and (x["cy_token_length"] <= max_length)
    )
    ds = ds.map(lambda _, idx: {"idx": idx}, with_indices=True)
    return ds


def get_tokenizer(ds: DatasetDict, config: dict[str, Any]) -> PreTrainedTokenizerFast:

    tokenizer_config = config["tokenizer"]
    run_path = utils.get_run_path(config)
    tokenizer_path = f"{run_path}/tokenizer"

    if os.path.isdir(tokenizer_path):
        return load_tokenizer(run_path)

    tokenizer = create_tokenizer(ds, tokenizer_config)
    save_tokenizer(tokenizer, run_path)

    return tokenizer
