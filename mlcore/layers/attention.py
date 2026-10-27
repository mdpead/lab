import torch
from torch import Tensor, nn

from mlcore.kernels import flash_attention
from mlcore.layers.position import RotaryPositionalEncoding


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        d_h: int,
        rope_layer: "RotaryPositionalEncoding",
    ) -> None:
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_h = d_h
        self.qkv_proj = nn.Linear(d_model, (num_heads + 2 * num_kv_heads) * d_h, bias=False)
        self.o_proj = nn.Linear(num_heads * d_h, d_model, bias=False)

        self.rope = rope_layer

    def forward(
        self, x: Tensor, seq_starts: Tensor, kv_cache: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:

        # Create Q, K, V tensors in a fused manner
        qkv = self.qkv_proj(x)  # (batch, seq, (num_heads + 2 * num_kv_heads) * d_h)

        qkv = qkv.reshape(
            qkv.shape[0], qkv.shape[1], self.num_heads + 2 * self.num_kv_heads, self.d_h
        )  # (batch, seq, num_heads + 2 * num_kv_heads, d_h)

        q, k_new, v_new = torch.split(
            qkv, [self.num_heads, self.num_kv_heads, self.num_kv_heads], dim=2
        )

        q = q.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, d_h)
        k_new = k_new.permute(0, 2, 1, 3)  # (batch, num_kv_heads, seq_len, d_h)
        v_new = v_new.permute(0, 2, 1, 3)  # (batch, num_kv_heads, seq_len, d_h)

        # Apply RoPE
        if kv_cache is not None:
            offset = kv_cache.shape[-2]
        else:
            offset = 0
        q, k_new = self.rope(q, k_new, offset)

        # Append onto the kv cache if using
        if kv_cache is not None:
            k = torch.concat((kv_cache[0], k_new), dim=2)
            v = torch.concat((kv_cache[1], v_new), dim=2)
        else:
            k = k_new
            v = v_new

        # K and V stay at num_kv_heads; the kernel maps each query head to its group
        attn_mh = flash_attention(q, k, v, seq_starts, causal=True)  # (batch, num_heads, seq, d_h)

        # Project out
        attn_flat = attn_mh.permute(0, 2, 1, 3).reshape(attn_mh.shape[0], attn_mh.shape[2], -1)
        O = self.o_proj(attn_flat)  # (batch, seq_len, d_model)

        # Wrap up the new kv cache entries; training has no cache to append them to
        kv_new = None if kv_cache is None else torch.stack((k_new, v_new), dim=0)

        return O, kv_new
