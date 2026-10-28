import itertools
from typing import Any

from mlcore import runs
from mlcore import tokenizer as tokenizer_utils
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
    return tokenizer_utils.get_tokenizer(ds, config, create_tokenizer)
