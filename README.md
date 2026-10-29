# lab

[![CI](https://github.com/mdpead/lab/actions/workflows/ci.yml/badge.svg)](https://github.com/mdpead/lab/actions/workflows/ci.yml)

Language models built from scratch in PyTorch, with no training framework: a translation model
that outperforms a model 25 times its size, and a reasoning model trained through pretraining,
fine-tuning and reinforcement learning on a single 16 GB GPU. The parts they share live in
`mlcore`, a small library of layers, Triton kernels and training infrastructure.

## Projects

### [English → Welsh translation](projects/en_cy_translation/)

A 50M-parameter encoder-decoder Transformer trained on the Cardiff University translation
memory, about 1.3M sentence pairs. It scores higher
than Meta's NLLB-200 (1.3B parameters) on all three test sets, including FLORES+, which was
checked for overlap with the training data.

| BLEU | FLORES+ (Wikipedia) | Cardiff TM (institutional) | Tatoeba (casual) |
|---|---|---|---|
| **This model (~50M)** | **46.3** | **57.5** | **47.4** |
| NLLB-200 distilled (1.3B) | 44.7 | 44.8 | 45.1 |
| NLLB-200 distilled (600M) | 36.1 | 37.1 | 39.1 |
| Opus-MT (74M) | 13.9 | 11.1 | 29.9 |

Published on Hugging Face as
[mdpead/en-cy-translation](https://huggingface.co/mdpead/en-cy-translation), usable through
`transformers.pipeline`.

### [Python execution reasoning](projects/python_reasoning/)

A 352M-parameter mixture-of-experts model (213M active per token) asked what generated Python
functions return. It writes out each executed line with its values, then the answer. Trained in
three stages on one RTX 5070 Ti: pretraining on 1.3B tokens of Python, supervised fine-tuning on
execution traces, and GRPO with rewards checked by running the code.

- **89.3%** greedy pass@1 on 3,022 held-out prompts.
- **Pretraining adds 2.9 points** (95% interval +1.7 to +4.2, bootstrapped over functions),
  but only at a high fine-tuning learning rate: at a third of the rate the same weights
  finish 8 points lower.
- **RL helped on one task of three**: about 5 points on a search task with a graded reward.
  On the two all-or-nothing tasks it moved accuracy by about a point, within the noise.

The [results](projects/python_reasoning/docs/results.md) and
[lessons](projects/python_reasoning/docs/lessons.md) write-ups cover what did not work as well
as what did.

## Built from scratch

- **Flash attention in Triton**, forward and backward. It handles grouped-query attention
  without expanding keys and values, and left-padded batches without a mask tensor. On causal
  attention it is 15–45% faster than PyTorch's fused `scaled_dot_product_attention` in forward
  timings at sequence lengths 128 to 1024
  ([details](projects/python_reasoning/docs/model.md#the-attention-kernel)).
- **Mixture of experts** with top-k routing, fixed expert capacity for static shapes during
  training, and a load-balancing loss.
- **Both Transformer families**: decoder-only (grouped-query attention, RoPE, RMSNorm, SwiGLU)
  and the original encoder-decoder.
- **Generation** with a KV cache, batched greedy and sampled decoding.
- **GRPO**, the reinforcement learning loop, with rewards from executing the model's answers.
- **Training infrastructure**: exact resume to the minibatch, seeding one stage from another's
  checkpoint, and config fingerprints that clear a stage when a setting it depends on changes.

About 300 tests. CI runs the CPU tests on every change; those that build a model or call a
Triton kernel need a GPU and run locally.

## mlcore

[mlcore/README.md](mlcore/README.md) explains how the library is organised: where the line
between library and project falls, the extension points a project plugs into, the config
contract, and how each extraction was checked to change nothing.

| Module | Contents |
|---|---|
| `mlcore.layers` | Grouped-query and multi-head attention, RoPE and sinusoidal positions, RMSNorm and LayerNorm, ReLU and SwiGLU feed-forwards, mixture of experts, embedding |
| `mlcore.models` | `decoder_only` and `encoder_decoder` Transformers, each with a `build_transformer` |
| `mlcore.kernels` | Triton flash attention, plus reference add and matmul kernels |
| `mlcore.generation` | Batched greedy and sampled decoding with a KV cache |
| `mlcore.training` | `supervised_loop`, which a project gives its loss; checkpoints, resume, stage seeding, LR schedules |
| `mlcore.data.samplers` | Resumable infinite and token-budget samplers |
| `mlcore.runs` | Run and stage directories, config fingerprints, logging |
| `mlcore.tokenizer` | Saving, loading and caching a trained tokenizer |
| `mlcore.testing` | Pytest fixtures and markers shared by the test suites |

## Running it

Requires Python 3.13. Training the reasoning model and running the Triton kernels needs Linux
and an NVIDIA GPU. Each project runs from its own folder, and its `requirements.txt` installs
`mlcore` from the repo root:

```bash
cd projects/python_reasoning
pip install -r requirements.txt
pytest
```

Each project's README has its training and evaluation commands. To work on `mlcore` itself,
run `pip install -r requirements.txt` and `pytest` from the repo root.

## Next

- Define models and training stages in config, with components chosen from a registry.
- Reasoning: harder program tiers, and GRPO with clipping and a KL penalty.

## License

Code is MIT-licensed: see [LICENSE](LICENSE). The trained translation model is CC BY 4.0,
following its training data; see [its README](projects/en_cy_translation/README.md#license).
