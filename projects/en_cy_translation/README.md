---
language:
  - en
  - cy
tags:
  - translation
  - en-cy
datasets:
  - techiaith/cardiff-university-tm-en-cy
  - openlanguagedata/flores_plus
  - agentlans/tatoeba-english-translations
license: cc-by-4.0
---

# English → Welsh Neural Machine Translation

[![CI](https://github.com/mdpead/en-cy-translation/actions/workflows/ci.yml/badge.svg)](https://github.com/mdpead/en-cy-translation/actions/workflows/ci.yml)

A Transformer-based sequence-to-sequence model for English to Welsh translation, implemented from scratch in PyTorch. Full training code available at [github.com/mdpead/en-cy-translation](https://github.com/mdpead/en-cy-translation).

## Usage

```python
from transformers import pipeline

pipe = pipeline("translation", model="mdpead/en-cy-translation")
pipe("Hello, how are you?")
```

Or with the model and tokenizer directly:

```python
from src.hf_wrapper import EnCyForTranslation
from transformers import PreTrainedTokenizerFast

model = EnCyForTranslation.from_pretrained("mdpead/en-cy-translation")
tokenizer = PreTrainedTokenizerFast.from_pretrained("mdpead/en-cy-translation")

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
output_ids = model.generate(**inputs, max_length=256)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

## Architecture

| Component | Detail |
|---|---|
| Model | Encoder-Decoder Transformer |
| Embedding dim (`d_model`) | 512 |
| Attention heads | 8 |
| Encoder / Decoder layers | 6 / 6 |
| Feed-forward dim (`d_ff`) | 2048 |
| Vocabulary size | 16,000 |
| Max sequence length | 256 tokens |
| Tokenizer | WordPiece (shared bilingual vocabulary) |

Training uses mixed-precision (AMP), gradient accumulation, and a warmup inverse square-root learning rate schedule.

## Training

- **Dataset**: [techiaith/cardiff-university-tm-en-cy](https://huggingface.co/datasets/techiaith/cardiff-university-tm-en-cy) (~1.3M sentence pairs)
- **Steps**: 50,000
- **Effective batch size**: 25,000 tokens
- **Optimiser**: AdamW (β₁=0.9, β₂=0.98, ε=1e-9)
- **Learning rate**: 1e-3 with 2,000 warmup steps, inverse square root decay

## Benchmark

Evaluated with beam search (beam size 4) against three publicly available EN→CY models across three datasets. NLLB-200 1.3B was run in fp16; all other models in fp32.

- **FLORES+**: [`openlanguagedata/flores_plus`](https://huggingface.co/datasets/openlanguagedata/flores_plus) devtest split (1012 sentences, Wikipedia text)
- **Cardiff**: [`techiaith/cardiff-university-tm-en-cy`](https://huggingface.co/datasets/techiaith/cardiff-university-tm-en-cy) 10% held-out test split (1000 sentences, institutional text)
- **Tatoeba**: [`agentlans/tatoeba-english-translations`](https://huggingface.co/datasets/agentlans/tatoeba-english-translations) Welsh subset (1613 sentences, casual/short)

FLORES+ devtest has been checked against the Cardiff training split on both the English and Welsh sides: no exact or normalised (case, punctuation and whitespace) matches, and no FLORES+ sentence shares 50% or more of its 8-grams with the training data. Reproduce with `python scripts/check_overlap.py --config base`.

### FLORES+ (out-of-distribution, Wikipedia)

| Model | Params | BLEU | spBLEU | chrF | chrF++ |
|-------|--------|------|--------|------|--------|
| **en-cy-translation** | ~50M | **46.29** | **51.89** | **68.39** | **66.38** |
| NLLB-200 (distilled 1.3B) | 1.3B | 44.65 | 48.16 | 66.10 | 64.15 |
| NLLB-200 (distilled 600M) | 600M | 36.14 | 37.57 | 58.61 | 56.55 |
| Opus-MT | 74M | 13.94 | 13.74 | 33.82 | 32.25 |

### Cardiff University TM (in-distribution, institutional)

| Model | BLEU | spBLEU | chrF | chrF++ |
|-------|------|--------|------|--------|
| **en-cy-translation** | **57.49** | **63.87** | **76.77** | **74.64** |
| NLLB-200 (distilled 1.3B) | 44.83 | 49.40 | 67.73 | 65.16 |
| NLLB-200 (distilled 600M) | 37.10 | 39.51 | 61.36 | 58.69 |
| Opus-MT | 11.05 | 10.69 | 31.19 | 29.17 |

### Tatoeba (casual, short sentences)

| Model | BLEU | spBLEU | chrF | chrF++ |
|-------|------|--------|------|--------|
| **en-cy-translation** | **47.37** | **50.21** | **66.57** | **64.22** |
| NLLB-200 (distilled 1.3B) | 45.13 | 46.59 | 63.93 | 61.80 |
| NLLB-200 (distilled 600M) | 39.12 | 39.15 | 58.21 | 56.17 |
| Opus-MT | 29.86 | 30.11 | 48.25 | 46.72 |

## Setup

This project lives in the [lab](../../README.md) monorepo and builds on its shared `mlcore`
library: the Transformer is `mlcore.models.encoder_decoder`, and checkpointing, resume and
run directories come from `mlcore.training` and `mlcore.runs`. Python 3.13; run pip from this
folder so `mlcore` installs from the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python -m scripts.train --config base
```

Checkpoints are saved to `models/<run-name>/translation/checkpoints/` every `checkpoint_steps`
steps, and training resumes automatically from the latest one. Changing a setting that the
checkpoints depend on (the model, tokenizer, or the stage's training and data settings) clears
the stage and starts again.

```bash
pytest                                  # tests, CPU only
ruff check . && ruff format --check .   # lint and formatting
```

## Project Structure

```
├── configs/          # YAML training configs
├── scripts/
│   ├── train.py      # Training entry point
│   ├── check_overlap.py  # FLORES+ / training data overlap check
│   └── push_to_hub.py
└── src/
    ├── tokenizer.py  # WordPiece tokenizer
    ├── datasets.py   # Dataset loading
    ├── dataloader.py # Collation and token-budget DataLoaders
    ├── train.py      # Training loop and BLEU validation
    ├── generation.py # Autoregressive decoding
    └── hf_wrapper.py # HuggingFace PreTrainedModel wrapper
```

## License

CC BY 4.0 — derived from the [Cardiff University Translation Memory](https://huggingface.co/datasets/techiaith/cardiff-university-tm-en-cy) dataset, also licensed CC BY 4.0. Attribution to Cardiff University Language Technologies Unit.
