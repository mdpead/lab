"""The schema for configs/<name>.yaml, checked when a script loads its config."""

from dataclasses import dataclass
from typing import Annotated

from lab.config import Component, resumable
from lab.models import MODELS, encoder_decoder
from lab.runs import RunConfig
from lab.tokenizer import TOKENIZERS
from lab.training import StageConfig, SupervisedStage


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
    tokenizer: Annotated[Component, TOKENIZERS.only("wordpiece")]
    stages: Stages
