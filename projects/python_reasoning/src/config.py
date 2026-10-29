"""The schema for configs/<name>.yaml, checked when a script loads its config."""

from dataclasses import dataclass
from typing import Literal

from mlcore.config import resumable
from mlcore.models import decoder_only
from mlcore.runs import RunConfig
from mlcore.training import SupervisedStage, TrainStage

Tier = Literal["easy", "medium", "hard"]
Task = Literal["output", "input", "search"]


@dataclass(frozen=True, kw_only=True)
class TokenizerConfig:
    vocab_size: int
    training_size: int
    digit_split: bool = False


@dataclass(frozen=True, kw_only=True)
class LMStage(SupervisedStage):
    """Pretraining and SFT: the supervised loop with the MoE load-balancing loss."""

    router_aux_loss_coef: float
    num_workers: int = resumable(0)
    prefetch_factor: int = resumable(2)


@dataclass(frozen=True, kw_only=True)
class RewardOptions:
    """Keyword arguments for the graded rewards."""

    partial_credit: Literal["match", "closest"] = "match"


@dataclass(frozen=True, kw_only=True)
class RLStage(TrainStage):
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
class Stages:
    pretrain: LMStage | None = None
    sft: LMStage | None = None
    rl: RLStage | None = None


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
class Data:
    pretrain: PretrainData | None = None
    sft: SFTData | None = None
    rl: SynthData | None = None


@dataclass(frozen=True, kw_only=True)
class Config(RunConfig):
    model: decoder_only.ModelConfig
    tokenizer: TokenizerConfig
    train: Stages
    data: Data
