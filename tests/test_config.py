"""Checking configs against their schemas: types, missing and unknown keys, resumable fields."""

from dataclasses import dataclass, field
from typing import Annotated, Literal

import pytest
import yaml

from lab.config import (
    Component,
    ConfigError,
    Registry,
    field_type,
    parse,
    resumable,
    resumable_keys,
    stage_class,
)
from lab.training import StageConfig, SupervisedStage, TrainStage


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
class _SFT(StageConfig):
    train: SupervisedStage


@dataclass(frozen=True, kw_only=True)
class _Stages:
    sft: _SFT | None = None


@dataclass(frozen=True, kw_only=True)
class _Schema:
    stages: _Stages


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
        "num_workers",
        "prefetch_factor",
    }


# --- components checked against their constructors ---

THINGS = Registry("thing", provided=("d_model",))


@THINGS.register("plain")
class _Plain:
    def __init__(self, d_model: int, width: int, scale: float = 0.5) -> None:
        self.d_model, self.width, self.scale = d_model, width, scale


@THINGS.register("checked")
class _Checked:
    def __init__(self, d_model: int, low: int, high: int) -> None:
        self.low, self.high = low, high

    @staticmethod
    def check_config(low: int, high: int) -> None:
        if low > high:
            raise ValueError(f"low ({low}) must not exceed high ({high})")


@dataclass(frozen=True, kw_only=True)
class _Holder:
    thing: Annotated[Component, THINGS]
    plain_only: Annotated[Component, THINGS.only("plain")] | None = None


def test_component_settings_come_from_the_constructor_with_its_defaults():
    holder = parse(_Holder, {"thing": {"type": "plain", "width": 3}})
    assert holder.thing == Component("plain", _Plain, {"width": 3, "scale": 0.5})
    built = holder.thing.build(d_model=8, unused=1)
    assert (built.d_model, built.width, built.scale) == (8, 3, 0.5)


def test_an_explicit_default_is_the_same_component():
    explicit = parse(_Holder, {"thing": {"type": "plain", "width": 3, "scale": 0.5}})
    assert explicit == parse(_Holder, {"thing": {"type": "plain", "width": 3}})


@pytest.mark.parametrize(
    "block, message",
    [
        (
            {"type": "plain", "widht": 3},
            r"unknown key 'widht' for thing 'plain'; did you mean 'width'",
        ),
        ({"type": "plain", "width": 3, "d_model": 8}, r"'d_model' is set by the model"),
        ({"type": "plain"}, r"thing: missing required key 'width'"),
        ({"type": "plain", "width": 2.5}, r"thing\.width: expected an integer"),
        ({"type": "plian", "width": 3}, r"unknown thing type 'plian'; did you mean 'plain'"),
        ({"type": "checked", "low": 3, "high": 1}, r"thing: low \(3\) must not exceed high \(1\)"),
    ],
)
def test_component_blocks_are_checked_when_the_config_loads(block, message):
    with pytest.raises(ConfigError, match=message):
        parse(_Holder, {"thing": block})


def test_only_restricts_a_field_to_some_names():
    data = {
        "thing": {"type": "plain", "width": 1},
        "plain_only": {"type": "checked", "low": 1, "high": 2},
    }
    with pytest.raises(
        ConfigError,
        match=r"plain_only: thing type 'checked' is not supported here \(one of: plain\)",
    ):
        parse(_Holder, data)


def test_component_builds_a_component_in_code():
    assert THINGS.component("plain", width=4) == Component(
        "plain", _Plain, {"width": 4, "scale": 0.5}
    )
    with pytest.raises(ConfigError, match="missing required key 'width'"):
        THINGS.component("plain")
