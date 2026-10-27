# lab

My machine learning workspace: a shared library of from-scratch PyTorch building blocks, and
the projects built on it.

## Layout

```
lab/
  mlcore/        shared library: layers, kernels, training loop, generation, config (planned)
  projects/
    en_cy_translation/   English → Welsh neural machine translation
    python_reasoning/    a language model trained to trace and predict Python execution
```

Each project keeps its own README, configs, scripts and tests. Run a project from its folder:

```bash
cd projects/python_reasoning
pip install -r requirements.txt
pytest
```

## Projects

| Project | Summary | Previously |
|---|---|---|
| [en_cy_translation](projects/en_cy_translation/) | Encoder-decoder Transformer for English → Welsh translation | [mdpead/en-cy-translation](https://github.com/mdpead/en-cy-translation) |
| [python_reasoning](projects/python_reasoning/) | Decoder-only MoE model, pretrained on Python, then SFT and RL on execution traces | [mdpead/python-reasoning-lm](https://github.com/mdpead/python-reasoning-lm) |

Both projects were brought in with their full commit history.

## Plan

1. Bring both projects in unchanged.
2. Extract the shared code (layers, kernels, training loop, generation, run management) into
   `mlcore`, starting from `python_reasoning`.
3. Move `en_cy_translation` onto `mlcore` and the newer dependencies.
4. Define models and training stages in config, with components chosen from a registry.
