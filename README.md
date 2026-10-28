# lab

My machine learning workspace: a shared library of from-scratch PyTorch building blocks, and
the projects built on it.

## Layout

```
lab/
  mlcore/        shared library: layers, models, kernels, training, generation, samplers
  tests/         mlcore's tests
  projects/
    en_cy_translation/   English → Welsh neural machine translation
    python_reasoning/    a language model trained to trace and predict Python execution
```

Each project keeps its own README, configs, scripts and tests. Run a project from its folder;
its `requirements.txt` installs `mlcore` from the repo root in editable mode:

```bash
cd projects/python_reasoning
pip install -r requirements.txt
pytest
```

To work on `mlcore` itself, from the repo root:

```bash
pip install -r requirements.txt
pytest                       # tests that build a model or call a Triton kernel need CUDA
```

## mlcore

| Module | Contents |
|---|---|
| `mlcore.layers` | Grouped-query and multi-head attention, RoPE and sinusoidal positions, RMSNorm and LayerNorm, ReLU and SwiGLU feed-forwards, mixture of experts, embedding |
| `mlcore.models.decoder_only` | The decoder-only Transformer and `build_transformer` |
| `mlcore.models.encoder_decoder` | The post-norm encoder-decoder Transformer and `build_transformer` |
| `mlcore.kernels` | Triton flash attention (forward and backward), plus reference add and matmul kernels |
| `mlcore.generation` | Batched greedy and sampled decoding with a KV cache |
| `mlcore.training` | Checkpoints, exact resume, seeding a stage from another, LR schedules, `train()` around a project's loop |
| `mlcore.data.samplers` | Resumable infinite and token-budget samplers |
| `mlcore.runs` | Run and stage directories, config fingerprints that clear stale stages, logging |
| `mlcore.tokenizer` | Saving, loading and caching a trained tokenizer |

## Projects

| Project | Summary | Previously |
|---|---|---|
| [en_cy_translation](projects/en_cy_translation/) | Encoder-decoder Transformer for English → Welsh translation | [mdpead/en-cy-translation](https://github.com/mdpead/en-cy-translation) |
| [python_reasoning](projects/python_reasoning/) | Decoder-only MoE model, pretrained on Python, then SFT and RL on execution traces | [mdpead/python-reasoning-lm](https://github.com/mdpead/python-reasoning-lm) |

Both projects were brought in with their full commit history.

## Plan

1. Bring both projects in unchanged.
2. Extract the shared code (layers, kernels, training loop, generation, run management) into
   `mlcore`, starting from `python_reasoning`. **Done**: moved without changing behaviour.
3. Move `en_cy_translation` onto `mlcore` and the newer dependencies. **Done**: the published
   model gives the same translations.
4. Define models and training stages in config, with components chosen from a registry.
