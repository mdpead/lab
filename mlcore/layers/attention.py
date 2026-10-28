import math

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


class MultiHeadAttention(nn.Module):
    """Plain multi-head attention with an explicit boolean mask, for self- and cross-attention.

    Masked entries are False in `attn_mask` (batch, q_len, k_len).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_h = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor) -> Tensor:
        # Create Q,K,V tensors
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        # Split into multi heads (batch_no, num_heads, seq_no, d_h)
        Q_mh = Q.reshape(Q.shape[0], Q.shape[1], self.num_heads, self.d_h).permute(0, 2, 1, 3)
        K_mh = K.reshape(K.shape[0], K.shape[1], self.num_heads, self.d_h).permute(0, 2, 1, 3)
        V_mh = V.reshape(V.shape[0], V.shape[1], self.num_heads, self.d_h).permute(0, 2, 1, 3)

        # Calculate attention matrices
        attn_raw = torch.matmul(Q_mh, K_mh.transpose(-1, -2)) / math.sqrt(self.d_h)
        attn_mask_mh = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        fill_value = torch.finfo(attn_raw.dtype).min  # works for fp16/bf16/fp32
        attn_masked = attn_raw.masked_fill(~attn_mask_mh, fill_value)
        attn = torch.softmax(attn_masked, -1)
        attn = self.dropout(attn)

        # Apply to value vectors and recombine
        A_mh = torch.matmul(attn, V_mh)
        A = A_mh.permute(0, 2, 1, 3).reshape(A_mh.shape[0], A_mh.shape[2], -1)

        # Project out
        O = self.W_o(A)

        return O
