"""Models built from config match the code before named components, weight for weight.

tests/fixtures/model_reference.json was recorded from that code: small models of each family,
built with a fixed seed. Names, shapes and the hash of every initialised tensor must match,
so checkpoints and published models load and seeded runs start from the same weights.
"""

import hashlib
import json

import pytest
import torch

from mlcore.config import ConfigError
from mlcore.layers import MixtureOfExpertsFeedForward, SwiGLUFeedForward
from mlcore.models import build_model, parse_model

REFERENCE = json.load(open("tests/fixtures/model_reference.json"))["tiny"]

# The models the reference was recorded from, as configs describe them now
MODELS = {
    "decoder_only": (
        {
            "type": "decoder_only",
            "d_model": 64,
            "num_layers": 2,
            "max_length": 64,
            "dropout": 0.0,
            "norm": {"type": "rms"},
            "position": {"type": "rope"},
            "attention": {"type": "gqa", "num_heads": 4, "num_kv_heads": 2, "d_h": 16},
            "feedforward": {
                "type": "moe",
                "d_ff": 128,
                "num_experts": 4,
                "top_k": 2,
                "capacity_factor": 4.0,
            },
        },
        100,
    ),
    "encoder_decoder": (
        {
            "type": "encoder_decoder",
            "d_model": 32,
            "num_enc_layers": 2,
            "num_dec_layers": 2,
            "max_length": 32,
            "dropout": 0.1,
            "norm": {"type": "layer"},
            "position": {"type": "sinusoidal"},
            "attention": {"type": "mha", "num_heads": 4},
            "feedforward": {"type": "relu", "d_ff": 64, "bias": True},
        },
        200,
    ),
}


def _config(family, **changes):
    model, vocab_size = MODELS[family]
    return {"model": {**model, **changes}, "tokenizer": {"vocab_size": vocab_size}}


def _hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


@pytest.mark.parametrize("family", sorted(MODELS))
def test_seeded_model_matches_the_reference_weight_for_weight(family):
    torch.manual_seed(0)
    state = build_model(_config(family)).state_dict()
    expected = REFERENCE[family]
    assert {k: list(v.shape) for k, v in state.items()} == expected["params"]
    assert {k: _hash(v) for k, v in state.items()} == expected["hashes"]


def test_unknown_component_type_suggests_the_closest():
    model = _config("decoder_only", norm={"type": "rsm"})["model"]
    with pytest.raises(
        ConfigError, match=r"model\.norm: unknown norm type 'rsm'; did you mean 'rms'"
    ):
        parse_model(model)


def test_component_block_needs_a_type():
    model = _config("decoder_only", norm={"eps": 1e-5})["model"]
    with pytest.raises(ConfigError, match=r"model\.norm: missing 'type' \(one of: rms, layer\)"):
        parse_model(model)


def test_component_settings_are_checked_against_its_class():
    attention = {"type": "gqa", "num_heads": 4, "num_kv_heads": 3, "d_h": 16}
    with pytest.raises(ConfigError, match=r"model\.attention: num_heads \(4\) must be a multiple"):
        parse_model(_config("decoder_only", attention=attention)["model"])


def test_a_dense_feedforward_can_replace_the_mixture_of_experts():
    model = build_model(_config("decoder_only", feedforward={"type": "swiglu", "d_ff": 96}))
    layer = model.decoder.decoder_layers[0]
    assert isinstance(layer.feedforward, SwiGLUFeedForward)
    assert not layer.feedforward_has_aux
    assert not any(isinstance(m, MixtureOfExpertsFeedForward) for m in model.modules())


@pytest.mark.cuda
def test_a_dense_feedforward_runs_with_zero_auxiliary_loss():
    model = build_model(_config("decoder_only", feedforward={"type": "swiglu", "d_ff": 96}))
    model = model.to("cuda").to(torch.float16).eval()
    ids = torch.randint(0, 100, (2, 8), device="cuda")
    logits, loss_aux = model(ids)
    assert logits.shape == (2, 8, 100)
    assert loss_aux.item() == 0.0
