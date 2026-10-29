"""Typed configs: check a parsed YAML dict against dataclasses before anything runs.

A project describes its config as frozen, keyword-only dataclasses (keyword-only, so a subclass
can add required fields after a base class's defaults). `parse` walks the YAML against them and
fails on the first problem, naming its path: a misspelt or unknown key, a missing setting, or
a value of the wrong type. Fields that can change without invalidating a stage's checkpoints
are marked with `resumable()`, and the stage fingerprint in `mlcore.runs` reads that mark.

Swappable components come from a `Registry`. A field annotated `Annotated[T, REGISTRY]` takes a
block like `{type: rms, eps: 1.0e-6}`: `type` picks a registered class, and the rest of the
block is checked against it. A layer class is checked against its own constructor, so its
settings are written down once. `REGISTRY.only(...)` narrows which names a field accepts.
"""

import dataclasses
import difflib
import inspect
import types
import typing
from collections.abc import Callable
from typing import Annotated, Any, Literal, Union

RESUMABLE = "resumable"

# A parsed config file, as the code reads it
Config = dict[str, Any]


class ConfigError(ValueError):
    """A config that does not match its schema. The message starts with the offending path."""


def resumable(default: Any = dataclasses.MISSING) -> Any:
    """A field that can change mid-run without invalidating the stage's checkpoints."""
    return dataclasses.field(default=default, metadata={RESUMABLE: True})


def resumable_keys(cls: type) -> frozenset[str]:
    """The names of a dataclass's fields marked `resumable()`."""
    return frozenset(f.name for f in dataclasses.fields(cls) if f.metadata.get(RESUMABLE))


class Registry:
    """Named implementations of one kind of component.

    An entry is either a config dataclass, checked field by field, or a class such as a layer,
    checked against its constructor: the arguments the model passes in (`provided`) are left
    out, and the rest are the block's settings, with the constructor's defaults and type hints.
    A class may define `check_config(**settings)`, raising ValueError, for checks across
    settings that should fail when the config loads rather than when the model is built.
    """

    def __init__(self, kind: str, provided: tuple[str, ...] = ()) -> None:
        self.kind = kind
        self.provided = frozenset(provided)
        self._classes: dict[str, type] = {}

    def register[T: type](self, name: str) -> Callable[[T], T]:
        def decorator(cls: T) -> T:
            if name in self._classes:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._classes[name] = cls
            return cls

        return decorator

    def names(self) -> list[str]:
        return list(self._classes)

    def get(self, name: str) -> type:
        return self._classes[name]

    def only(self, *names: str) -> "Only":
        """This registry, restricted to `names`, for a field that accepts only some of them."""
        unknown = set(names) - set(self._classes)
        if unknown:
            raise ValueError(f"no {self.kind} registered as {sorted(unknown)}")
        return Only(self, names)

    def component(self, name: str, **settings: Any) -> Any:
        """The parsed form of `{type: name, **settings}`, for building configs in code."""
        return _convert_registered(self, self.names(), {"type": name, **settings}, self.kind)


@dataclasses.dataclass(frozen=True)
class Only:
    registry: Registry
    names: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Component:
    """A component a config chose: its registered class and its checked settings."""

    name: str
    cls: type
    settings: dict[str, Any]

    def build(self, **provided: Any) -> Any:
        """Construct it, passing each of `provided` that its constructor takes."""
        accepted = inspect.signature(self.cls).parameters
        return self.cls(**{k: v for k, v in provided.items() if k in accepted}, **self.settings)

    def __getitem__(self, key: str) -> Any:
        return self.settings[key]


def field_type(cls: type, name: str) -> type:
    """The declared type of a dataclass field, with `| None` and `Annotated` removed."""
    hints = typing.get_type_hints(cls)
    if name not in hints:
        raise ConfigError(f"{cls.__name__} has no field {name!r}")
    hint = hints[name]
    if _is_union(hint):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return hint


def field_hint(cls: type, name: str) -> Any:
    """The declared type of a dataclass field as written, `Annotated` registries included."""
    hints = typing.get_type_hints(cls, include_extras=True)
    if name not in hints:
        raise ConfigError(f"{cls.__name__} has no field {name!r}")
    return hints[name]


def canonical(value: Any) -> Any:
    """A comparable form of a parsed config, leaving out fields marked `resumable()`.

    Dataclasses keep their class name, so two components with the same settings but of
    different types (an RMSNorm and a LayerNorm with the same eps) still differ.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple(
                (f.name, canonical(getattr(value, f.name)))
                for f in dataclasses.fields(value)
                if not f.metadata.get(RESUMABLE)
            ),
        )
    if isinstance(value, dict):
        return tuple(sorted((str(k), canonical(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(canonical(v) for v in value)
    return value


def stage_class(schema: type, stage: str) -> type:
    """The dataclass describing `stages.<stage>.train` in a project's config schema."""
    return field_type(field_type(field_type(schema, "stages"), stage), "train")


def parse[T](cls: type[T], data: Any, source: str | None = None, path: str = "") -> T:
    """Build `cls` from a parsed YAML mapping, or raise ConfigError naming what is wrong.

    Errors give the dotted path to the problem, after `source` (such as the file name).
    `path` is where `data` sits in the whole config, when parsing just one section.
    """
    try:
        return _convert(cls, data, path)
    except ConfigError as e:
        if source is None:
            raise
        raise ConfigError(f"{source}: {e}") from e


def _join(path: str, key: Any) -> str:
    return f"{path}.{key}" if path else str(key)


def _where(path: str) -> str:
    return path or "top level"


def _is_union(hint: Any) -> bool:
    return typing.get_origin(hint) in (Union, types.UnionType)


def _describe(value: Any) -> str:
    return f"{type(value).__name__} {value!r}"


def _convert(hint: Any, value: Any, path: str) -> Any:
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        return _convert_dataclass(hint, value, path)

    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    if origin is Annotated:
        for meta in hint.__metadata__:
            if isinstance(meta, Registry):
                return _convert_registered(meta, meta.names(), value, path)
            if isinstance(meta, Only):
                return _convert_registered(meta.registry, list(meta.names), value, path)
        return _convert(args[0], value, path)
    if hint is Any:
        return value
    if hint is type(None):
        if value is not None:
            raise ConfigError(f"{_where(path)}: expected null, got {_describe(value)}")
        return None
    if _is_union(hint):
        options = [a for a in args if a is not type(None)]
        if value is None and len(options) < len(args):
            return None
        if len(options) == 1:
            # `X | None` with a value: report X's own error, not a list of failed options
            return _convert(options[0], value, path)
        errors = []
        for arg in args:
            try:
                return _convert(arg, value, path)
            except ConfigError as e:
                errors.append(str(e).removeprefix(f"{_where(path)}: "))
        raise ConfigError(f"{_where(path)}: no option matched ({'; '.join(errors)})")
    if origin is Literal:
        if value not in args:
            options = ", ".join(repr(a) for a in args)
            raise ConfigError(f"{_where(path)}: expected one of {options}, got {value!r}")
        return value
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{_where(path)}: expected a list, got {_describe(value)}")
        return [_convert(args[0], v, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is tuple:
        # YAML has no tuples, so a list stands in for one
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{_where(path)}: expected a list, got {_describe(value)}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{_where(path)}: expected {len(args)} items, got {len(value)}")
        return tuple(_convert(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{_where(path)}: expected a mapping, got {_describe(value)}")
        return {
            _convert(args[0], k, _join(path, "<key>")): _convert(args[1], v, _join(path, k))
            for k, v in value.items()
        }
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{_where(path)}: expected true or false, got {_describe(value)}")
        return value
    if hint is int:
        # bool is a subclass of int, but `true` for a count is a mistake
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"{_where(path)}: expected an integer, got {_describe(value)}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            hint_text = ""
            if isinstance(value, str):
                # PyYAML reads 3e-4 as a string; YAML 1.1 floats need a dot: 3.0e-4
                hint_text = " (YAML needs a dot in exponent floats, e.g. 3.0e-4)"
            raise ConfigError(
                f"{_where(path)}: expected a number, got {_describe(value)}{hint_text}"
            )
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{_where(path)}: expected a string, got {_describe(value)}")
        return value
    raise TypeError(f"{_where(path)}: unsupported type in schema: {hint!r}")


def _convert_dataclass(cls: type, value: Any, path: str) -> Any:
    if not isinstance(value, dict):
        raise ConfigError(f"{_where(path)}: expected a mapping, got {_describe(value)}")

    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = [k for k in value if k not in fields]
    if unknown:
        key = unknown[0]
        close = difflib.get_close_matches(str(key), fields, n=1)
        suggestion = f"; did you mean {close[0]!r}?" if close else ""
        raise ConfigError(
            f"{_where(path)}: unknown key {key!r}{suggestion} (valid keys: {', '.join(fields)})"
        )

    hints = typing.get_type_hints(cls, include_extras=True)
    kwargs = {}
    for name, f in fields.items():
        if name in value:
            kwargs[name] = _convert(hints[name], value[name], _join(path, name))
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{_where(path)}: missing required key {name!r}")
    try:
        return cls(**kwargs)
    except ValueError as e:
        # A __post_init__ check across fields, reported at the section it applies to
        raise ConfigError(f"{_where(path)}: {e}") from e


def _convert_registered(registry: Registry, allowed: list[str], value: Any, path: str) -> Any:
    options = ", ".join(allowed)
    if not isinstance(value, dict):
        raise ConfigError(
            f"{_where(path)}: expected a mapping with a 'type', got {_describe(value)}"
        )
    if "type" not in value:
        raise ConfigError(f"{_where(path)}: missing 'type' (one of: {options})")

    name = value["type"]
    if name not in allowed:
        if name in registry.names():
            raise ConfigError(
                f"{_where(path)}: {registry.kind} type {name!r} is not supported here "
                f"(one of: {options})"
            )
        close = difflib.get_close_matches(str(name), allowed, n=1)
        suggestion = f"; did you mean {close[0]!r}?" if close else ""
        raise ConfigError(
            f"{_where(path)}: unknown {registry.kind} type {name!r}{suggestion} (one of: {options})"
        )

    cls = registry.get(name)
    settings = {k: v for k, v in value.items() if k != "type"}
    if dataclasses.is_dataclass(cls):
        return _convert_dataclass(cls, settings, path)
    return _convert_component(registry, name, cls, settings, path)


def _convert_component(
    registry: Registry, name: str, cls: type, settings: dict[str, Any], path: str
) -> Component:
    """Check a component block against its class's constructor."""
    hints = typing.get_type_hints(cls.__init__)
    params = {
        p.name: p
        for p in inspect.signature(cls).parameters.values()
        if p.name not in registry.provided
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    for key in settings:
        if key in registry.provided:
            raise ConfigError(f"{_where(path)}: {key!r} is set by the model, not in the config")
        if key not in params:
            close = difflib.get_close_matches(str(key), params, n=1)
            suggestion = f"; did you mean {close[0]!r}?" if close else ""
            valid = ", ".join(params) or "none"
            raise ConfigError(
                f"{_where(path)}: unknown key {key!r} for {registry.kind} {name!r}{suggestion} "
                f"(valid keys: {valid})"
            )

    checked = {}
    for key, param in params.items():
        if key in settings:
            checked[key] = _convert(hints[key], settings[key], _join(path, key))
        elif param.default is inspect.Parameter.empty:
            raise ConfigError(f"{_where(path)}: missing required key {key!r}")
        else:
            checked[key] = param.default

    check = getattr(cls, "check_config", None)
    if check is not None:
        try:
            check(**checked)
        except ValueError as e:
            raise ConfigError(f"{_where(path)}: {e}") from e
    return Component(name, cls, checked)
