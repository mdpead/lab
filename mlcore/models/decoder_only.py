import torch
from torch import Tensor, nn

from mlcore.layers import (
    Embedding,
    GroupedQueryAttention,
    MixtureOfExpertsFeedForward,
    RMSNorm,
    RotaryPositionalEncoding,
)
from mlcore.runs import Config


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        d_h: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float,
        dropout: float,
        rope_layer: RotaryPositionalEncoding,
    ) -> None:
        super().__init__()
        self.attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads, d_h, rope_layer)
        self.feedforward = MixtureOfExpertsFeedForward(
            d_model, d_ff, num_experts, top_k, capacity_factor
        )
        self.layer_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(0, 2)])
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
        ff_out, loss_aux = self.feedforward(self.layer_norms[1](x), padding_mask)
        x = x + self.dropout(ff_out)
        return x, loss_aux, new_kv


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        d_h: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float,
        num_dec_layers: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super().__init__()
        self.rope_layer = RotaryPositionalEncoding(d_h, max_length)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    num_heads,
                    num_kv_heads,
                    d_h,
                    d_ff,
                    num_experts,
                    top_k,
                    capacity_factor,
                    dropout,
                    self.rope_layer,
                )
                for _ in range(num_dec_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)

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


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        d_h: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float,
        num_layers: int,
        vocab_size: int,
        max_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_length = max_length  # context bound: RoPE table size and training seq length
        self.num_kv_heads = num_kv_heads
        self.d_h = d_h
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, d_model)
        self.decoder = Decoder(
            d_model,
            num_heads,
            num_kv_heads,
            d_h,
            d_ff,
            num_experts,
            top_k,
            capacity_factor,
            num_layers,
            dropout,
            max_length,
        )
        self.dropout = nn.Dropout(dropout)

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


def build_transformer(config: Config) -> Transformer:

    model_config = config["model"]

    transformer = Transformer(
        model_config["d_model"],
        model_config["num_heads"],
        model_config["num_kv_heads"],
        model_config["d_h"],
        model_config["d_ff"],
        model_config["num_experts"],
        model_config["top_k"],
        model_config["capacity_factor"],
        model_config["num_layers"],
        config["tokenizer"]["vocab_size"],
        model_config["max_length"],
        model_config["dropout"],
    )
    transformer = transformer.apply(init_weights)
    return transformer
