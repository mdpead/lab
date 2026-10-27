import math

import torch

from src import model


def test_linear_matches_reference():
    layer = model.Linear(4, 3)
    x = torch.randn(2, 4)
    assert torch.allclose(layer(x), x @ layer.weight.t() + layer.bias)


def test_layer_norm_normalises_last_dim():
    x = torch.randn(3, 5, 16) * 10 + 4
    out = model.LayerNorm(16)(x)
    assert torch.allclose(out.mean(-1), torch.zeros(3, 5), atol=1e-5)
    assert torch.allclose(out.std(-1, unbiased=False), torch.ones(3, 5), atol=1e-4)


def test_positional_encoding_values():
    pe = model.PositionalEncoding(d_model=8, max_length=10).PE
    assert pe.shape == (10, 8)
    assert torch.equal(pe[0, 0::2], torch.zeros(4))
    assert torch.equal(pe[0, 1::2], torch.ones(4))
    assert math.isclose(pe[3, 2].item(), math.sin(3 / 10000 ** (2 / 8)), rel_tol=1e-5)
    assert math.isclose(pe[3, 3].item(), math.cos(3 / 10000 ** (2 / 8)), rel_tol=1e-5)


def test_embedding_is_scaled():
    emb = model.Embedding(10, 16)
    assert torch.allclose(emb(torch.tensor([[3]]))[0, 0], emb.E[3] * 4)


def test_attention_ignores_masked_keys():
    torch.manual_seed(0)
    attn = model.MultiHeadAttention(16, 4, dropout=0.0)
    q = torch.randn(1, 3, 16)
    kv = torch.randn(1, 5, 16)
    mask = torch.tensor([[True, True, True, False, False]]).expand(3, -1).unsqueeze(0)

    out = attn(q, kv, kv, mask)
    kv_changed = kv.clone()
    kv_changed[:, 3:] = torch.randn(1, 2, 16)

    assert out.shape == (1, 3, 16)
    assert torch.allclose(out, attn(q, kv_changed, kv_changed, mask), atol=1e-6)


def test_decoder_self_attn_mask_is_causal_and_padded():
    padding = torch.tensor([[True, True, False]])
    mask = model.Decoder(16, 4, 32, 1, 0.0).make_self_attn_mask(padding, padding)
    assert mask.tolist() == [[[True, False, False], [True, True, False], [False, False, False]]]


def test_cross_attn_mask_hides_source_padding():
    tgt = torch.tensor([[True, True]])
    src = torch.tensor([[True, False, False]])
    mask = model.Decoder(16, 4, 32, 1, 0.0).make_cross_attn_mask(tgt, src)
    assert mask.tolist() == [[[True, False, False], [True, False, False]]]


def test_build_transformer_output_shape(transformer, config):
    src = torch.randint(5, 50, (2, 7))
    tgt = torch.randint(5, 50, (2, 4))
    out = transformer(
        src, tgt, torch.ones(2, 7, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool)
    )
    assert out.shape == (2, 4, config["tokenizer"]["vocab_size"])


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
