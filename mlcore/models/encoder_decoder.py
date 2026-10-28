"""Post-norm encoder-decoder Transformer, as in "Attention Is All You Need".

Attribute names are part of the saved weights, published models included: renaming one
breaks loading them.
"""

import torch
from torch import Tensor, nn

from mlcore.layers import (
    Embedding,
    LayerNorm,
    MultiHeadAttention,
    PositionWiseFeedForward,
    SinusoidalPositionalEncoding,
)
from mlcore.runs import Config


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feedforward = PositionWiseFeedForward(d_model, d_ff, bias=True)
        self.layer_norms = nn.ModuleList([LayerNorm(d_model) for _ in range(0, 2)])
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
        self, d_model: int, num_heads: int, d_ff: int, num_enc_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_enc_layers)]
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
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feedforward = PositionWiseFeedForward(d_model, d_ff, bias=True)
        self.layer_norms = nn.ModuleList([LayerNorm(d_model) for _ in range(0, 3)])
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
        self, d_model: int, num_heads: int, d_ff: int, num_dec_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_dec_layers)]
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


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_enc_layers: int,
        num_dec_layers: int,
        vocab_size: int,
        max_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.src_embedding = Embedding(vocab_size, d_model)
        self.tgt_embedding = Embedding(vocab_size, d_model)
        self.positional_encoder = SinusoidalPositionalEncoding(d_model, max_length)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_enc_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_dec_layers, dropout)
        self.output = Output(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

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


def build_transformer(config: Config) -> Transformer:

    model_config = config["model"]

    transformer = Transformer(
        model_config["d_model"],
        model_config["num_heads"],
        model_config["d_ff"],
        model_config["num_enc_layers"],
        model_config["num_dec_layers"],
        config["tokenizer"]["vocab_size"],
        model_config["max_length"],
        model_config["dropout"],
    )
    transformer = transformer.apply(init_weights)
    return transformer
