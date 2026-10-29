"""Model families, chosen by the config's `model.type`."""

from typing import Annotated, Any

from torch import nn

from mlcore.config import Config, parse
from mlcore.models import decoder_only, encoder_decoder
from mlcore.models.registry import MODELS

__all__ = ["MODELS", "build_model", "parse_model", "upgrade_legacy_model"]


def parse_model(model: dict[str, Any], base: type = object) -> Any:
    """The typed config for a `model` section. `base` restricts which families are allowed."""
    return parse(Annotated[base, MODELS], model, path="model")


def build_model(config: Config, base: type = object) -> nn.Module:
    """Build the model a config describes, with freshly initialised weights."""
    return parse_model(config["model"], base).build(config["tokenizer"]["vocab_size"])


def upgrade_legacy_model(model: dict[str, Any]) -> dict[str, Any]:
    """A `model` section in today's form, from either form a config may have been saved in.

    Sections saved before components were named have no `type` and list flat sizes; which
    family they describe follows from their keys. Anything else is returned unchanged.
    """
    if "type" in model:
        return model
    if "num_enc_layers" in model:
        return encoder_decoder.upgrade_legacy(model)
    if "num_kv_heads" in model:
        return decoder_only.upgrade_legacy(model)
    return model
