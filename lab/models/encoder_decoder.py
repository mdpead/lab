"""Post-norm encoder-decoder Transformer, as in "Attention Is All You Need".

Attribute names are part of the saved weights, published models included: renaming one
breaks loading them.
"""

from typing import Annotated, Any

import torch
from torch import Tensor, nn

from lab.config import Component
from lab.layers import ATTENTIONS, FEEDFORWARDS, NORMS, POSITIONS, Embedding
from lab.models.registry import MODELS


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        norm: Component,
        attention: Component,
        feedforward: Component,
    ) -> None:
        super().__init__()
        self.attention = attention.build(d_model=d_model, dropout=dropout)
        self.feedforward = feedforward.build(d_model=d_model)
        self.layer_norms = nn.ModuleList([norm.build(d_model=d_model) for _ in range(0, 2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor, attn_mask: Tensor) -> Tensor:
        x = src
        attn_out = self.attention(x, x, x, attn_mask)
        x = self.layer_norms[0](x + self.dropout(attn_out))
        ff_out = self.feedforward(x)
        x = self.layer_norms[1](x + self.dropout(ff_out))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        dropout: float,
        norm: Component,
        attention: Component,
        feedforward: Component,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, dropout, norm, attention, feedforward)
                for _ in range(num_layers)
            ]
        )

    def make_attn_mask(self, q_padding_mask: Tensor, k_padding_mask: Tensor) -> Tensor:
        padding_mask = q_padding_mask.unsqueeze(-1) & k_padding_mask.unsqueeze(-2)
        return padding_mask

    def forward(self, src: Tensor, src_padding_mask: Tensor) -> Tensor:
        attn_mask = self.make_attn_mask(src_padding_mask, src_padding_mask)
        for layer in self.layers:
            src = layer(src, attn_mask)
        return src


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        norm: Component,
        attention: Component,
        feedforward: Component,
    ) -> None:
        super().__init__()
        self.self_attention = attention.build(d_model=d_model, dropout=dropout)
        self.cross_attention = attention.build(d_model=d_model, dropout=dropout)
        self.feedforward = feedforward.build(d_model=d_model)
        self.layer_norms = nn.ModuleList([norm.build(d_model=d_model) for _ in range(0, 3)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        self_attn_mask: Tensor,
        cross_attn_mask: Tensor,
    ) -> Tensor:
        x = tgt
        self_attn_out = self.self_attention(tgt, tgt, tgt, self_attn_mask)
        x = self.layer_norms[0](x + self.dropout(self_attn_out))
        cross_attn_out = self.cross_attention(x, src, src, cross_attn_mask)
        x = self.layer_norms[1](x + self.dropout(cross_attn_out))
        ff_out = self.feedforward(x)
        x = self.layer_norms[2](x + self.dropout(ff_out))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        dropout: float,
        norm: Component,
        attention: Component,
        feedforward: Component,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, dropout, norm, attention, feedforward)
                for _ in range(num_layers)
            ]
        )

    def make_self_attn_mask(self, q_padding_mask: Tensor, k_padding_mask: Tensor) -> Tensor:
        padding_mask = q_padding_mask.unsqueeze(-1) & k_padding_mask.unsqueeze(-2)
        causal_mask = torch.tril(
            torch.ones(
                q_padding_mask.shape[1], k_padding_mask.shape[1], device=q_padding_mask.device
            ).bool()
        ).unsqueeze(0)
        self_attn_mask = causal_mask & padding_mask
        return self_attn_mask

    def make_cross_attn_mask(self, q_padding_mask: Tensor, k_padding_mask: Tensor) -> Tensor:
        cross_attn_mask = q_padding_mask.unsqueeze(-1) & k_padding_mask.unsqueeze(-2)
        return cross_attn_mask

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
    ) -> Tensor:
        self_attn_mask = self.make_self_attn_mask(tgt_padding_mask, tgt_padding_mask)
        cross_attn_mask = self.make_cross_attn_mask(tgt_padding_mask, src_padding_mask)

        for layer in self.layers:
            tgt = layer(src, tgt, self_attn_mask, cross_attn_mask)
        return tgt


class Output(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_dec: Tensor) -> Tensor:
        x = self.linear(tgt_dec)
        return x


@MODELS.register("encoder_decoder")
class Transformer(nn.Module):
    """A post-norm encoder-decoder Transformer. Its constructor is the config's `model`
    section, and the vocabulary size comes from `tokenizer.vocab_size`."""

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int,
        num_enc_layers: int,
        num_dec_layers: int,
        max_length: int,
        dropout: float,
        norm: Annotated[Component, NORMS],
        position: Annotated[Component, POSITIONS.only("sinusoidal")],
        attention: Annotated[Component, ATTENTIONS.only("mha")],
        # A mixture of experts returns an auxiliary loss, which this model has no place for
        feedforward: Annotated[Component, FEEDFORWARDS.only("relu", "swiglu")],
    ) -> None:
        super().__init__()
        self.src_embedding = Embedding(vocab_size, d_model)
        self.tgt_embedding = Embedding(vocab_size, d_model)
        self.positional_encoder = position.build(d_model=d_model, max_length=max_length)
        self.encoder = Encoder(d_model, num_enc_layers, dropout, norm, attention, feedforward)
        self.decoder = Decoder(d_model, num_dec_layers, dropout, norm, attention, feedforward)
        self.output = Output(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.apply(init_weights)

    @staticmethod
    def check_config(d_model: int, attention: Component, **_: Any) -> None:
        if d_model % attention["num_heads"]:
            raise ValueError(
                f"d_model ({d_model}) must be a multiple of attention.num_heads "
                f"({attention['num_heads']})"
            )

    def encode(self, src: Tensor, src_padding_mask: Tensor) -> Tensor:
        src_emb = self.dropout(self.positional_encoder(self.src_embedding(src)))
        src_enc = self.encoder(src_emb, src_padding_mask)
        return src_enc

    def decode(
        self,
        src_enc: Tensor,
        tgt: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
    ) -> Tensor:
        tgt_emb = self.dropout(self.positional_encoder(self.tgt_embedding(tgt)))
        tgt_dec = self.decoder(src_enc, tgt_emb, src_padding_mask, tgt_padding_mask)
        return tgt_dec

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
    ) -> Tensor:
        src_enc = self.encode(src, src_padding_mask)
        tgt_dec = self.decode(src_enc, tgt, src_padding_mask, tgt_padding_mask)
        out = self.output(tgt_dec)
        return out


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Embedding, Embedding)):
        nn.init.normal_(m.E, mean=0.0, std=0.02)
