"""The encoder-decoder Transformer and its layers. Runs on CPU."""

import math

import pytest
import torch

from mlcore.config import ConfigError
from mlcore.layers import (
    Embedding,
    LayerNorm,
    MultiHeadAttention,
    PositionWiseFeedForward,
    SinusoidalPositionalEncoding,
)
from mlcore.models import build_model, encoder_decoder, parse_model

CONFIG = {
    "model": {
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
    "tokenizer": {"vocab_size": 60},
}


@pytest.fixture
def transformer():
    torch.manual_seed(0)
    return build_model(CONFIG)


def test_layer_norm_normalises_last_dim():
    x = torch.randn(3, 5, 16) * 10 + 4
    out = LayerNorm(16)(x)
    assert torch.allclose(out.mean(-1), torch.zeros(3, 5), atol=1e-5)
    assert torch.allclose(out.std(-1, unbiased=False), torch.ones(3, 5), atol=1e-4)


def test_positional_encoding_values():
    pe = SinusoidalPositionalEncoding(d_model=8, max_length=10).PE
    assert pe.shape == (10, 8)
    assert torch.equal(pe[0, 0::2], torch.zeros(4))
    assert torch.equal(pe[0, 1::2], torch.ones(4))
    assert math.isclose(pe[3, 2].item(), math.sin(3 / 10000 ** (2 / 8)), rel_tol=1e-5)
    assert math.isclose(pe[3, 3].item(), math.cos(3 / 10000 ** (2 / 8)), rel_tol=1e-5)


def test_positional_encoding_is_saved_with_the_weights():
    assert "PE" in SinusoidalPositionalEncoding(8, 10).state_dict()


def test_embedding_is_scaled():
    emb = Embedding(10, 16)
    assert torch.allclose(emb(torch.tensor([[3]]))[0, 0], emb.E[3] * 4)


def test_feedforward_bias_is_optional():
    assert PositionWiseFeedForward(8, 16).linear1.bias is None
    assert PositionWiseFeedForward(8, 16, bias=True).linear1.bias is not None


def test_attention_ignores_masked_keys():
    torch.manual_seed(0)
    attn = MultiHeadAttention(16, 4, dropout=0.0)
    q = torch.randn(1, 3, 16)
    kv = torch.randn(1, 5, 16)
    mask = torch.tensor([[True, True, True, False, False]]).expand(3, -1).unsqueeze(0)

    out = attn(q, kv, kv, mask)
    kv_changed = kv.clone()
    kv_changed[:, 3:] = torch.randn(1, 2, 16)

    assert out.shape == (1, 3, 16)
    assert torch.allclose(out, attn(q, kv_changed, kv_changed, mask), atol=1e-6)


def _small_config():
    return parse_model({**CONFIG["model"], "d_model": 16, "num_dec_layers": 1})


def test_decoder_self_attn_mask_is_causal_and_padded():
    padding = torch.tensor([[True, True, False]])
    mask = encoder_decoder.Decoder(_small_config()).make_self_attn_mask(padding, padding)
    assert mask.tolist() == [[[True, False, False], [True, True, False], [False, False, False]]]


def test_cross_attn_mask_hides_source_padding():
    tgt = torch.tensor([[True, True]])
    src = torch.tensor([[True, False, False]])
    mask = encoder_decoder.Decoder(_small_config()).make_cross_attn_mask(tgt, src)
    assert mask.tolist() == [[[True, False, False], [True, False, False]]]


def test_build_model_output_shape(transformer):
    src = torch.randint(5, 50, (2, 7))
    tgt = torch.randint(5, 50, (2, 4))
    out = transformer(
        src, tgt, torch.ones(2, 7, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool)
    )
    assert out.shape == (2, 4, CONFIG["tokenizer"]["vocab_size"])


def test_decoder_is_causal(transformer):
    transformer.eval()
    src = torch.randint(5, 50, (1, 6))
    tgt = torch.randint(5, 50, (1, 5))
    ones = torch.ones(1, 6, dtype=torch.bool), torch.ones(1, 5, dtype=torch.bool)
    tgt_changed = tgt.clone()
    tgt_changed[0, 3:] = torch.randint(5, 50, (2,))

    out = transformer(src, tgt, *ones)
    out_changed = transformer(src, tgt_changed, *ones)

    assert torch.allclose(out[:, :3], out_changed[:, :3], atol=1e-5)
    assert not torch.allclose(out[:, 3:], out_changed[:, 3:], atol=1e-5)


def test_source_padding_does_not_change_output(transformer):
    transformer.eval()
    src = torch.randint(5, 50, (1, 6))
    tgt = torch.randint(5, 50, (1, 4))
    tgt_mask = torch.ones(1, 4, dtype=torch.bool)
    src_padded = torch.cat([src, torch.full((1, 3), 2)], dim=1)
    src_padded_mask = torch.tensor([[True] * 6 + [False] * 3])

    out = transformer(src, tgt, torch.ones(1, 6, dtype=torch.bool), tgt_mask)
    out_padded = transformer(src_padded, tgt, src_padded_mask, tgt_mask)

    assert torch.allclose(out, out_padded, atol=1e-5)


def test_can_overfit_single_example(transformer):
    torch.manual_seed(0)
    src = torch.randint(5, 50, (1, 6))
    tgt = torch.randint(5, 50, (1, 6))
    masks = torch.ones(1, 6, dtype=torch.bool), torch.ones(1, 5, dtype=torch.bool)
    optimiser = torch.optim.Adam(transformer.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    transformer.train()
    for _ in range(100):
        logits = transformer(src, tgt[:, :-1], *masks)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    transformer.eval()
    assert torch.equal(transformer(src, tgt[:, :-1], *masks).argmax(-1), tgt[:, 1:])


def test_a_dense_feedforward_can_be_swapped_in():
    """Components are chosen by config: SwiGLU in place of the ReLU feed-forward."""
    torch.manual_seed(0)
    model = build_model(
        {**CONFIG, "model": {**CONFIG["model"], "feedforward": {"type": "swiglu", "d_ff": 64}}}
    )
    assert "encoder.layers.0.feedforward.gate.weight" in model.state_dict()
    src = torch.randint(5, 50, (2, 7))
    tgt = torch.randint(5, 50, (2, 4))
    masks = torch.ones(2, 7, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool)
    assert model(src, tgt, *masks).shape == (2, 4, 60)


def test_a_mixture_of_experts_is_rejected_for_the_encoder_decoder():
    moe = {"type": "moe", "d_ff": 64, "num_experts": 4, "top_k": 2, "capacity_factor": 1.0}
    message = r"model\.feedforward: feedforward type 'moe' is not supported here"
    with pytest.raises(ConfigError, match=message + r" \(one of: relu, swiglu\)"):
        parse_model({**CONFIG["model"], "feedforward": moe})
