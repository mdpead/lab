"""Checking configs against their schemas: types, missing and unknown keys, resumable fields."""

from dataclasses import dataclass, field
from typing import Literal

import pytest
import yaml

from mlcore.config import ConfigError, field_type, parse, resumable, resumable_keys, stage_class
from mlcore.training import SupervisedStage, TrainStage


@dataclass(frozen=True, kw_only=True)
class Inner:
    size: int
    rate: float = 0.5


@dataclass(frozen=True, kw_only=True)
class Outer:
    name: str
    flag: bool = False
    mode: Literal["a", "b"] = "a"
    pair: tuple[float, float] = (0.0, 0.0)
    items: list[int] = field(default_factory=list)
    inner: Inner | None = None
    steps: int = resumable(10)


def test_builds_nested_dataclasses_and_applies_defaults():
    cfg = parse(Outer, {"name": "x", "inner": {"size": 3}, "pair": [0.9, 0.95]})
    assert cfg == Outer(name="x", inner=Inner(size=3), pair=(0.9, 0.95))
    assert cfg.inner.rate == 0.5


def test_unknown_key_suggests_the_closest_valid_one():
    with pytest.raises(ConfigError, match=r"^inner: unknown key 'sise'; did you mean 'size'\?"):
        parse(Outer, {"name": "x", "inner": {"sise": 3}})


def test_missing_required_key_is_named_with_its_section():
    with pytest.raises(ConfigError, match=r"^inner: missing required key 'size'"):
        parse(Outer, {"name": "x", "inner": {}})
    with pytest.raises(ConfigError, match=r"^top level: missing required key 'name'"):
        parse(Outer, {})


@pytest.mark.parametrize(
    "data, message",
    [
        ({"name": 3}, r"^name: expected a string, got int 3"),
        ({"name": "x", "flag": 1}, r"^flag: expected true or false"),
        ({"name": "x", "steps": True}, r"^steps: expected an integer, got bool"),
        ({"name": "x", "steps": 2.5}, r"^steps: expected an integer, got float"),
        ({"name": "x", "mode": "c"}, r"^mode: expected one of 'a', 'b', got 'c'"),
        ({"name": "x", "pair": [1.0]}, r"^pair: expected 2 items, got 1"),
        ({"name": "x", "items": [1, "two"]}, r"^items\[1\]: expected an integer"),
        ({"name": "x", "inner": {"size": 1, "rate": "fast"}}, r"^inner\.rate: expected a number"),
    ],
)
def test_wrong_types_are_rejected_at_their_path(data, message):
    with pytest.raises(ConfigError, match=message):
        parse(Outer, data)


def test_an_integer_is_accepted_where_a_float_is_expected():
    assert parse(Inner, {"size": 1, "rate": 1}).rate == 1.0


def test_yaml_exponent_without_a_dot_gets_a_hint():
    """PyYAML reads 3e-4 as a string, which would otherwise surface far from the config."""
    data = yaml.safe_load("size: 1\nrate: 3e-4\n")
    assert data["rate"] == "3e-4"
    with pytest.raises(ConfigError, match=r"needs a dot in exponent floats"):
        parse(Inner, data)


def test_source_prefixes_the_message():
    with pytest.raises(ConfigError, match=r"^configs/x\.yaml: name: expected a string"):
        parse(Outer, {"name": 3}, "configs/x.yaml")


def test_resumable_keys_and_field_types():
    assert resumable_keys(Outer) == {"steps"}
    assert field_type(Outer, "inner") is Inner


@dataclass(frozen=True, kw_only=True)
class _Stages:
    sft: SupervisedStage | None = None


@dataclass(frozen=True, kw_only=True)
class _Schema:
    train: _Stages


def test_stage_class_finds_the_stage_schema():
    assert stage_class(_Schema, "sft") is SupervisedStage
    with pytest.raises(ConfigError, match="no field 'rl'"):
        stage_class(_Schema, "rl")


STAGE = {
    "device": "cpu",
    "learning_rate": 1e-3,
    "adam_betas": [0.9, 0.98],
    "adam_eps": 1e-9,
    "warm_up_steps": 10,
    "minibatch_token_size": 100,
    "num_steps": 50,
    "checkpoint_steps": 10,
    "validation_steps": 10,
    "validation_batches": 1,
}


def test_train_stage_defaults_match_what_the_training_code_assumes():
    stage = parse(TrainStage, STAGE)
    assert stage.lr_schedule == "inverse_sqrt"
    assert stage.label_smoothing == 0.0
    assert stage.keep_checkpoints is None
    assert stage.compile_model is False


def test_supervised_stage_needs_a_budget_of_at_least_one_minibatch():
    parse(SupervisedStage, {**STAGE, "effective_batch_token_size": 400})
    with pytest.raises(ConfigError, match=r"^top level: effective_batch_token_size \(50\)"):
        parse(SupervisedStage, {**STAGE, "effective_batch_token_size": 50})


def test_train_stage_resumable_keys():
    assert resumable_keys(SupervisedStage) == {
        "minibatch_token_size",
        "num_steps",
        "checkpoint_steps",
        "validation_steps",
        "validation_batches",
        "keep_checkpoints",
        "cache_clear_steps",
        "compile_model",
    }
