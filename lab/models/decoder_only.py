from typing import Annotated

import torch
from torch import Tensor, nn

from lab.config import Component
from lab.layers import (
    ATTENTIONS,
    FEEDFORWARDS,
    NORMS,
    POSITIONS,
    Embedding,
    MixtureOfExpertsFeedForward,
    RotaryPositionalEncoding,
)
from lab.models.registry import MODELS


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        norm: Component,
        attention: Component,
        feedforward: Component,
        rope_layer: RotaryPositionalEncoding,
    ) -> None:
        super().__init__()
        self.attention = attention.build(d_model=d_model, rope_layer=rope_layer)
        self.feedforward = feedforward.build(d_model=d_model)
        # A mixture of experts takes the padding mask and adds a load-balancing loss
        self.feedforward_has_aux = isinstance(self.feedforward, MixtureOfExpertsFeedForward)
        self.layer_norms = nn.ModuleList([norm.build(d_model=d_model) for _ in range(0, 2)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        seq_starts: Tensor,
        kv_cache: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        attn_out, new_kv = self.attention(self.layer_norms[0](x), seq_starts, kv_cache)
        x = x + self.dropout(attn_out)
        if self.feedforward_has_aux:
            ff_out, loss_aux = self.feedforward(self.layer_norms[1](x), padding_mask)
        else:
            ff_out, loss_aux = self.feedforward(self.layer_norms[1](x)), x.new_zeros(())
        x = x + self.dropout(ff_out)
        return x, loss_aux, new_kv


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        max_length: int,
        dropout: float,
        norm: Component,
        position: Component,
        attention: Component,
        feedforward: Component,
    ) -> None:
        super().__init__()
        self.rope_layer = position.build(d_h=attention["d_h"], max_length=max_length)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model, dropout, norm, attention, feedforward, self.rope_layer)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = norm.build(d_model=d_model)

    def forward(
        self,
        x: Tensor,
        seq_starts: Tensor,
        kv_cache: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        loss_aux_sum = 0
        new_kvs = []
        for i, layer in enumerate(self.decoder_layers):
            layer_cache = None if kv_cache is None else kv_cache[i]
            x, loss_aux, new_kv = layer(x, seq_starts, layer_cache, padding_mask)
            loss_aux_sum += loss_aux
            new_kvs.append(new_kv)

        loss_aux = loss_aux_sum / len(self.decoder_layers)
        x = self.final_norm(x)
        new_kv = None if kv_cache is None else torch.stack(new_kvs)  # (layers, 2, b, kv_h, q, d_h)
        return x, loss_aux, new_kv


@MODELS.register("decoder_only")
class Transformer(nn.Module):
    """A pre-norm decoder-only Transformer. Its constructor is the config's `model` section,
    and the vocabulary size comes from `tokenizer.vocab_size`."""

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int,
        num_layers: int,
        max_length: int,
        dropout: float,
        norm: Annotated[Component, NORMS],
        position: Annotated[Component, POSITIONS.only("rope")],
        attention: Annotated[Component, ATTENTIONS.only("gqa")],
        feedforward: Annotated[Component, FEEDFORWARDS],
    ) -> None:
        super().__init__()
        # context bound: RoPE table size and training seq length
        self.max_length = max_length
        self.num_kv_heads = attention["num_kv_heads"]
        self.d_h = attention["d_h"]
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, d_model)
        self.decoder = Decoder(
            d_model, num_layers, max_length, dropout, norm, position, attention, feedforward
        )
        self.dropout = nn.Dropout(dropout)
        self.apply(init_weights)

    def forward(
        self,
        x: Tensor,
        seq_starts: Tensor | None = None,
        kv_cache: Tensor | None = None,
        padding_mask: Tensor | None = None,
        output_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor | None]:
        """`output_mask` is (batch, seq) bool: project only those positions to the vocabulary.

        The logits then come back as (selected, vocab) in row-major order, which avoids
        the full (batch, seq, vocab) tensor when only some positions are scored.
        """
        # Right-padded training needs no offsets; left-padded generation passes real ones
        if seq_starts is None:
            seq_starts = torch.zeros(x.shape[0], dtype=torch.int32, device=x.device)

        x_emb = self.dropout(self.embedding(x))
        x_dec, loss_aux, new_kv = self.decoder(x_emb, seq_starts, kv_cache, padding_mask)
        if output_mask is not None:
            x_dec = x_dec[output_mask]
        out = torch.matmul(x_dec, self.embedding.E.T)
        # Cache-free callers get a two-tuple
        if kv_cache is None:
            return out, loss_aux
        return out, loss_aux, new_kv


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Embedding, Embedding)):
        nn.init.normal_(m.E, mean=0.0, std=0.02)
