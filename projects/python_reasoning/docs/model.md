# Model and attention kernel

## Model

A decoder-only transformer. The `base` configuration is 352M parameters (213M active per token), 16 layers, width 1024.

- **Attention:** grouped-query attention (16 query heads sharing 4 key/value heads) with rotary position embeddings, computed by a custom flash-attention kernel.
- **Feed-forward:** a mixture of 4 SwiGLU experts with top-2 routing and a load-balancing loss. Experts have fixed capacity during training so tensor shapes stay static; inference never drops a token.
- **Normalisation:** RMSNorm, pre-norm.
- **Tokenizer:** byte-level BPE (32k), with every digit as its own token so arithmetic is compositional, and reserved reasoning markers.

### The attention kernel

[`lab/kernels/flash_attention.py`](../../../lab/kernels/flash_attention.py) is a flash attention written in [Triton](https://github.com/triton-lang/triton), forward and backward, used for both training and generation.

- Never materialises the full score matrix: it works in tiles with an online softmax.
- Understands grouped-query attention directly, so keys and values are never expanded to the query head count.
- Takes a per-row start offset, so left-padded batches need no attention mask tensor.
- Handles the cached-decode case in the forward pass, where the queries are only the last few positions of the key sequence.
- Autotuned over tile sizes, and checked against PyTorch's reference attention for values and gradients.

#### How it compares

Three ways to compute the same attention:

| | Basic attention | PyTorch built-in | This kernel |
|---|---|---|---|
| What it is | `softmax(Q Kᵀ / √d) V` written out directly | `scaled_dot_product_attention`, a fused C++/CUDA implementation | Flash attention written in Triton |
| Score matrix | Built in full: one `seq × seq` matrix per head | Not built, when its flash backend is used | Never built |
| Extra memory | Grows with the square of the sequence length | Grows linearly | Grows linearly |
| Grouped-query attention | Keys and values must be repeated per query head | Keys and values must be repeated per query head | Native: each query head reads its group's keys directly |
| Left-padded batches | Needs a mask tensor | Needs a mask tensor, which cannot be combined with its `is_causal` option | A start offset per row |

PyTorch's built-in is itself a flash attention where it can be: it chooses between a flash backend, a memory-efficient backend and a plain one, preferring flash when the inputs are eligible. That backend is enabled here and accepts the benchmark's inputs, so the comparison below is one flash attention against another, not flash against a naive baseline.

Forward-pass time for causal attention, the case the model uses, on an RTX 5070 Ti in fp16 with batch 4, 4 heads and head dimension 64:

| Sequence length | Basic | PyTorch built-in | This kernel | vs basic | vs built-in |
|---|---|---|---|---|---|
| 128 | 0.036 ms | 0.011 ms | 0.009 ms | 4.0x faster | 1.2x faster |
| 256 | 0.047 ms | 0.015 ms | 0.012 ms | 3.9x faster | 1.3x faster |
| 512 | 0.095 ms | 0.023 ms | 0.020 ms | 4.8x faster | 1.15x faster |
| 1024 | 0.339 ms | 0.062 ms | 0.043 ms | 8.0x faster | 1.45x faster |

The gap to basic attention widens with length, as expected when one side is building a matrix that grows with the square of it. Against the built-in the kernel is 15–45% faster on causal attention. On non-causal attention the two are level: the kernel is 1.1–1.4x faster at lengths 128 and 256 and 4–7% slower at 512 and 1024, and about 2.5–3x faster than basic attention at the longer lengths.

What these numbers do and do not show:

- They are small, forward-only timings with equal query and key/value head counts. The ratios to the built-in were stable across three runs; the basic implementation's timings had occasional large outliers at short lengths.
- They do not cover the backward pass, memory use, or the model's real shapes.
- They do not measure the two features in the table that the built-in lacks. Avoiding the repeat of keys and values under grouped-query attention, and avoiding a mask tensor for padded batches, are where the kernel should gain most in the actual model, and neither is benchmarked yet.

Reproduce with `pytest tests/test_flash_attention_perf.py --benchmark -s`, run from the repo root.

