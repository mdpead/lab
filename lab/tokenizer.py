"""Tokenizers: trained from a project's text, saved with the run, and loaded by later stages.

The `tokenizer` section of a config picks a registered type and its settings, checked against
the class's constructor like model components are. A project supplies only the text to train
on.
"""

import itertools
import os
from collections.abc import Callable, Iterable
from typing import Annotated, Any

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

from lab import runs
from lab.config import Registry, parse

TOKENIZERS = Registry("tokenizer")

# The text a tokenizer trains on, drawn from a project's datasets
Corpus = Callable[[Any], Iterable[str]]


def _special_tokens(
    bos: str, eos: str, pad: str, mask: str | None, unk: str, extra: Iterable[str], reserved: int
) -> list[str]:
    """Special tokens in id order: the roles, then the project's own, then spare slots."""
    roles = [bos, eos, pad, *([mask] if mask is not None else []), unk]
    return [*roles, *extra, *[f"<|reserved_{i}|>" for i in range(reserved)]]


def _wrap(
    tokenizer: Tokenizer, bos: str, eos: str, pad: str, unk: str, mask: str | None
) -> PreTrainedTokenizerFast:
    """Frame each sequence with bos and eos, and name the special-token roles."""
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{bos} $A {eos}",
        special_tokens=[(bos, tokenizer.token_to_id(bos)), (eos, tokenizer.token_to_id(eos))],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=bos,
        eos_token=eos,
        pad_token=pad,
        unk_token=unk,
        **({"mask_token": mask} if mask is not None else {}),
    )


@TOKENIZERS.register("wordpiece")
class WordPieceTokenizer:
    """WordPiece over whitespace-separated words, as in BERT.

    `strip_accents` decomposes characters and drops their accents before training, so "tân"
    and "tan" become the same word.
    """

    def __init__(
        self,
        vocab_size: int,
        training_size: int | None = None,
        strip_accents: bool = False,
        bos: str = "[BOS]",
        eos: str = "[EOS]",
        pad: str = "[PAD]",
        unk: str = "[UNK]",
        mask: str | None = "[MASK]",
        special_tokens: tuple[str, ...] = (),
        reserved_tokens: int = 0,
    ) -> None:
        self.vocab_size = vocab_size
        self.training_size = training_size
        self.strip_accents = strip_accents
        self.roles = {"bos": bos, "eos": eos, "pad": pad, "unk": unk, "mask": mask}
        self.special = _special_tokens(bos, eos, pad, mask, unk, special_tokens, reserved_tokens)

    def train(self, texts: Iterable[str]) -> PreTrainedTokenizerFast:
        tokenizer = Tokenizer(models.WordPiece(unk_token=self.roles["unk"]))
        if self.strip_accents:
            tokenizer.normalizer = normalizers.Sequence(
                [normalizers.NFD(), normalizers.StripAccents()]
            )
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        trainer = trainers.WordPieceTrainer(vocab_size=self.vocab_size, special_tokens=self.special)
        tokenizer.train_from_iterator(itertools.islice(texts, self.training_size), trainer)
        tokenizer.decoder = decoders.WordPiece(prefix="##")
        return _wrap(tokenizer, **self.roles)


@TOKENIZERS.register("bpe")
class ByteLevelBPETokenizer:
    """Byte-level BPE, as in GPT-2: any text encodes, with no unknown characters.

    `digit_split` gives every digit its own token, so numbers are composed digit by digit.
    """

    def __init__(
        self,
        vocab_size: int,
        training_size: int | None = None,
        digit_split: bool = False,
        bos: str = "<bos>",
        eos: str = "<eos>",
        pad: str = "<pad>",
        unk: str = "<unk>",
        mask: str | None = None,
        special_tokens: tuple[str, ...] = (),
        reserved_tokens: int = 0,
    ) -> None:
        self.vocab_size = vocab_size
        self.training_size = training_size
        self.digit_split = digit_split
        self.roles = {"bos": bos, "eos": eos, "pad": pad, "unk": unk, "mask": mask}
        self.special = _special_tokens(bos, eos, pad, mask, unk, special_tokens, reserved_tokens)

    def train(self, texts: Iterable[str]) -> PreTrainedTokenizerFast:
        tokenizer = Tokenizer(models.BPE(unk_token=self.roles["unk"]))
        byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False)
        if self.digit_split:
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [pre_tokenizers.Digits(individual_digits=True), byte_level]
            )
        else:
            tokenizer.pre_tokenizer = byte_level
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special)
        tokenizer.train_from_iterator(itertools.islice(texts, self.training_size), trainer)
        return _wrap(tokenizer, **self.roles)


def train_tokenizer(tokenizer_config: dict[str, Any], texts: Iterable[str]) -> Any:
    """Train the tokenizer a config's `tokenizer` section describes on `texts`."""
    component = parse(Annotated[Any, TOKENIZERS], tokenizer_config, path="tokenizer")
    return component.build().train(texts)


def save_tokenizer(tokenizer: PreTrainedTokenizerFast, dir: str) -> None:
    tokenizer.save_pretrained(f"{dir}/tokenizer")


def load_tokenizer(dir: str) -> PreTrainedTokenizerFast:
    if not os.path.isdir(f"{dir}/tokenizer"):
        raise FileNotFoundError(
            f"No tokenizer in {dir} — train one for this config first "
            "(the project's tokenizer step, or get_tokenizer)."
        )
    return PreTrainedTokenizerFast.from_pretrained(f"{dir}/tokenizer")


def get_tokenizer(ds: Any, config: runs.Config, corpus: Corpus) -> PreTrainedTokenizerFast:
    """The run's tokenizer: loaded if already trained, otherwise trained on `corpus(ds)`."""
    run_path = runs.get_run_path(config)
    if os.path.isdir(f"{run_path}/tokenizer"):
        return load_tokenizer(run_path)

    tokenizer = train_tokenizer(config["tokenizer"], corpus(ds))
    save_tokenizer(tokenizer, run_path)
    return tokenizer
