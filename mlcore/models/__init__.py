"""Model families, chosen by the config's `model.type`."""

from typing import Annotated, Any

from torch import nn

from mlcore.config import Config, parse

# Imported so each family registers itself with MODELS
from mlcore.models import decoder_only, encoder_decoder
from mlcore.models.registry import MODELS

__all__ = ["MODELS", "build_model", "decoder_only", "encoder_decoder", "parse_model"]


def parse_model(model: dict[str, Any], *families: str) -> Any:
    """The typed config for a `model` section, optionally restricted to some `families`."""
    registry = MODELS.only(*families) if families else MODELS
    return parse(Annotated[Any, registry], model, path="model")


def build_model(config: Config) -> nn.Module:
    """Build the model a config describes, with freshly initialised weights."""
    return parse_model(config["model"]).build(config["tokenizer"]["vocab_size"])
