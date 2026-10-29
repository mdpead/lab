"""The schema for configs/<name>.yaml, checked when a script loads its config."""

from dataclasses import dataclass
from typing import Annotated, Literal

from lab.config import Component
from lab.models import MODELS, decoder_only
from lab.runs import RunConfig
from lab.tokenizer import TOKENIZERS
from lab.training import StageConfig, SupervisedStage, TrainStage

Tier = Literal["easy", "medium", "hard"]
Task = Literal["output", "input", "search"]


@dataclass(frozen=True, kw_only=True)
class LMTrain(SupervisedStage):
    """Pretraining and SFT: the supervised loop with the MoE load-balancing loss."""

    router_aux_loss_coef: float


@dataclass(frozen=True, kw_only=True)
class RewardOptions:
    """Keyword arguments for the graded rewards."""

    partial_credit: Literal["match", "closest"] = "match"


@dataclass(frozen=True, kw_only=True)
class RLTrain(TrainStage):
    router_aux_loss_coef: float
    generation_prompts_size: int
    effective_prompts_size: int
    completions_per_prompt: int
    max_new_tokens: int
    temperature: float
    validation_prompts_size: int | None = None
    validation_samples: int | None = None
    advantage_sd_floor: float = 1e-8
    reward_options: RewardOptions | None = None


@dataclass(frozen=True, kw_only=True)
class PretrainData:
    test_split_ratio: float
    num_shards: int | None = None
    sample_size: int | None = None


@dataclass(frozen=True, kw_only=True)
class SynthData:
    """Generated functions: how many, how hard, and which tasks to format them as."""

    num_records: int
    tier: Tier
    tasks: list[Task]
    inputs_per_fn: int
    test_split_ratio: float


@dataclass(frozen=True, kw_only=True)
class SFTData(SynthData):
    with_trace: bool


@dataclass(frozen=True, kw_only=True)
class PretrainStage(StageConfig):
    """Pretraining on StarCoder. A config can give only `data`, the corpus the tokenizer is
    trained on, and no `train` section when it does not pretrain."""

    train: LMTrain | None = None
    data: PretrainData


@dataclass(frozen=True, kw_only=True)
class SFTStage(StageConfig):
    train: LMTrain
    data: SFTData


@dataclass(frozen=True, kw_only=True)
class RLStage(StageConfig):
    train: RLTrain
    data: SynthData


@dataclass(frozen=True, kw_only=True)
class Stages:
    pretrain: PretrainStage | None = None
    sft: SFTStage | None = None
    rl: RLStage | None = None


@dataclass(frozen=True, kw_only=True)
class Config(RunConfig):
    model: Annotated[decoder_only.ModelConfig, MODELS]
    tokenizer: Annotated[Component, TOKENIZERS.only("bpe")]
    stages: Stages
