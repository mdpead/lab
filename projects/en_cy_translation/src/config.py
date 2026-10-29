"""The schema for configs/<name>.yaml, checked when a script loads its config."""

from dataclasses import dataclass
from typing import Annotated

from mlcore.config import resumable
from mlcore.models import MODELS, encoder_decoder
from mlcore.runs import RunConfig
from mlcore.training import StageConfig, SupervisedStage


@dataclass(frozen=True, kw_only=True)
class TokenizerConfig:
    vocab_size: int
    pad_token_id: int
    # Sentences to train the tokenizer on; null uses the whole train split
    training_size: int | None = None


@dataclass(frozen=True, kw_only=True)
class TranslationTrain(SupervisedStage):
    num_workers: int = resumable(0)


@dataclass(frozen=True, kw_only=True)
class TranslationData:
    ds: str
    test_split_ratio: float
    sample_size: int | None = None


@dataclass(frozen=True, kw_only=True)
class TranslationStage(StageConfig):
    train: TranslationTrain
    data: TranslationData


@dataclass(frozen=True, kw_only=True)
class Stages:
    translation: TranslationStage


@dataclass(frozen=True, kw_only=True)
class Config(RunConfig):
    model: Annotated[encoder_decoder.ModelConfig, MODELS]
    tokenizer: TokenizerConfig
    stages: Stages
