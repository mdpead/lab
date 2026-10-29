# mlcore

The library the projects in this repo build on: model layers, Triton kernels, generation, and
the training infrastructure around a project's own loss. This page explains how it is
organised and why: where the line between library and project falls, how a project plugs in,
and how each extraction was checked to change nothing.

## Who uses what

| | `en_cy_translation` | `python_reasoning` |
|---|---|---|
| `models` | `encoder_decoder` | `decoder_only` |
| `layers`, `kernels` | through its model | through its model, including Triton flash attention |
| `training` | `train`, `supervised_loop` | `train`, `supervised_loop` for pretraining and SFT; `train` with its own RL loop |
| `data.samplers` | `TokenSampler` | `InfiniteRandomSampler`, `TokenSampler` |
| `generation` | | KV-cache decoding, for evaluation and RL rollouts |
| `runs`, `tokenizer` | yes | yes |

## Where the line falls

**The library owns mechanics; the project owns meaning.** The library runs the parts that are
the same whatever the task: gradient accumulation, mixed precision, clipping, the validation
and checkpoint schedule, exact resume, and run directories. The project decides what a batch
contains, what the loss is, what counts as a good result, and how its data and tokenizer are
built.

**Code moves in when a second project needs it, not before.** Nothing was designed in advance.
`mlcore` began as the parts of `python_reasoning` that `en_cy_translation` duplicated, and each
later move followed the same rule:

| Stayed in the project | Why |
|---|---|
| The RL loop (`python_reasoning/src/rl.py`) | One user, and it does not fit the supervised shape: it generates rollouts, scores them by running code, and trains on group-relative advantages. It still uses `training.train` for resume and checkpoints. |
| Encoder-decoder decoding (`en_cy_translation/src/generation.py`, greedy, for BLEU during training; beam search comes from its Hugging Face wrapper) | One user. `mlcore.generation` is the decoder-only KV-cache path the reasoning project needs. |
| Batch collation, data preparation, tokenizer training | Task-specific by nature: they decide what a batch means. |
| BLEU, execution-checked answers | The task's definition of success, passed to the library as a callback. |

**Moves change nothing.** Each extraction was a move, verified to leave results unchanged,
before any redesign. Interface changes are kept small and stated explicitly; see
[How each move was checked](#how-each-move-was-checked).

## Extension points

A project plugs in with plain functions rather than subclasses. There are five:

| Hook | Signature | Supplied to |
|---|---|---|
| `compute_loss` | `(model, batch) -> (loss, values_to_log)` | `training.supervised_loop` |
| `tokens_key` | the batch field whose non-pad tokens count towards throughput | `training.supervised_loop` |
| `validation_metrics` | `(model, batch) -> dict[str, float]`, optional | `training.supervised_loop` |
| `loop` | `(stage, model, dataloaders, tokenizer, run, config) -> None` | `training.train` |
| `create_tokenizer` | `(datasets, tokenizer_config) -> PreTrainedTokenizerFast` | `tokenizer.get_tokenizer` |

The translation project's whole training module is a loss, a BLEU callback and two calls into
`mlcore`:

```python
def compute_loss(model, batch, criterion):
    logits = model(batch["src_input_ids"], batch["tgt_input_ids"],
                   batch["src_padding_mask"], batch["tgt_padding_mask"])
    loss = criterion(logits.reshape(-1, logits.shape[2]), batch["tgt_output_ids"].reshape(-1))
    return loss, {"loss": loss}


def train_loop(stage, model, dataloaders, tokenizer, run, config):
    training.supervised_loop(
        stage, model, dataloaders, tokenizer, run, config,
        compute_loss=partial(compute_loss, criterion=run["criterion"]),
        tokens_key="src_input_ids",
        validation_metrics=lambda m, batch: generate_metrics(m, batch, ...),  # BLEU
    )


def train(model, dataloaders, tokenizer, config):
    return training.train(STAGE, model, dataloaders, tokenizer, config, loop=train_loop)
```

The reasoning project's loss adds the mixture-of-experts load-balancing term and logs both
parts; everything else is the same. Its RL stage passes its own `loop` to `training.train`
instead of using `supervised_loop`.

## A stage's lifecycle

A run has one or more stages (the reasoning project has pretrain, SFT and RL; translation has
one), each with its own directory:

```
models/<name>/
  config.json            the config the run was last started with
  tokenizer/             shared by every stage
  <stage>/
    config.json          the config this stage was last started with
    <stage>.log          timestamped log, appended across restarts
    results.json         every train and validation step's logged values
    checkpoints/<step>.pt
```

1. **`runs.init_run(config, stage)`** creates the directories and compares the stage's
   *fingerprint* with the one saved last time. If they differ, the stage's directory is
   cleared, so a stale checkpoint is never resumed under a new setting. A changed tokenizer
   section clears the saved tokenizer the same way.
2. **`training.train(...)`** builds the run state. It resumes from the stage's latest
   checkpoint if one exists. Otherwise it starts fresh, optionally seeding the weights from
   another stage's checkpoint (`init_from`, pinned with `init_from_step`), with a fresh
   optimiser and step counter. If `num_steps` is already reached it returns.
3. **The loop** fast-forwards the sampler to the exact minibatch it stopped at, without reading
   any data, then trains. It checkpoints every `checkpoint_steps` and once more at the end, and
   keeps the newest `keep_checkpoints`.

## The config contract

Projects use YAML configs, which `mlcore` reads as plain dicts. Which keys it reads is not yet
enforced by a schema (see [Limitations](#limitations-and-next-steps)), so this is the contract:

| Key | Read by |
|---|---|
| `name`, `seed`, `locations.models_dir` | `runs`: run directory and seeding |
| `tokenizer.vocab_size` | both model builders |
| `model.*` | the model builder: `d_model`, `num_heads`, `d_ff`, `max_length`, `dropout`, plus `num_kv_heads`, `d_h`, `num_experts`, `top_k`, `capacity_factor`, `num_layers` (decoder-only) or `num_enc_layers`, `num_dec_layers` (encoder-decoder) |
| `train.<stage>.*` | `training`: `device`, `learning_rate`, `adam_betas`, `adam_eps`, `warm_up_steps`, `lr_schedule`, `label_smoothing`, `effective_batch_token_size`, `minibatch_token_size`, `num_steps`, `checkpoint_steps`, `keep_checkpoints`, `validation_steps`, `validation_batches`, `cache_clear_steps`, `compile_model`, `init_from_step` |
| `data.<stage>` | `runs`, for the fingerprint only |

A stage's **fingerprint** is `name`, `seed`, `model`, `tokenizer`, `data.<stage>`, and
`train.<stage>` minus the keys that can change mid-run without invalidating a checkpoint:
`num_steps`, `checkpoint_steps`, `keep_checkpoints`, `validation_steps`, `validation_batches`,
`minibatch_token_size`, `cache_clear_steps`, `num_workers`, `prefetch_factor`, `compile_model`.
So raising `num_steps` extends a finished run, while changing the learning rate starts the
stage again.

## How each move was checked

| Change | Check |
|---|---|
| Extract the shared code from `python_reasoning` ([#2](https://github.com/mdpead/lab/pull/2)) | Every line of the original files compared against the new layout; the same 243 tests before and after, with the same 101 passing on CPU. |
| Add the encoder-decoder and move `en_cy_translation` onto `mlcore` and newer dependencies ([#3](https://github.com/mdpead/lab/pull/3)) | Weight names kept, so existing checkpoints and the published model load. The published model's greedy and beam-search translations identical across old and new code and versions, with logits within 1e-5. |
| Share the supervised loop ([#5](https://github.com/mdpead/lab/pull/5)) | Bitwise-identical `results.json`, apart from timings, before and after: translation on CPU and GPU, reasoning SFT on GPU. |
| Share the test fixtures ([#7](https://github.com/mdpead/lab/pull/7)) | The same tests collected and passing in all three suites before and after. |

## Testing

`mlcore`'s tests are in `tests/` at the repo root. Tests that build a decoder-only model or call
a Triton kernel are marked `cuda` and skip without a GPU; CI runs the rest on every change.
`mlcore.testing` is a pytest plugin with the shared markers and fixtures, loaded from a
conftest with `pytest_plugins = ["mlcore.testing"]`.

## Limitations and next steps

- **The config contract is implicit.** Configs are untyped dicts, so a misspelt key fails when
  it is first read, possibly hours into a run, and the fingerprint's key lists live in
  `runs.py` rather than with the settings they describe.
- **Some conventions are fixed.** `runs.parse_config` reads `configs/<name>.yaml` relative to
  the working directory. The optimiser is AdamW with weight decay 0.01, and gradients are
  clipped at norm 1.0.
- **`supervised_loop` takes the tokenizer only for its pad id.**
- **Generation is decoder-only.** The encoder-decoder's decoding stays in its project until a
  second encoder-decoder needs it.
- **Two projects is a small sample.** The boundaries fit both, but a third project would test
  whether they generalise.

The next step is typed config classes with a registry of components, so that models and
training stages are defined in config and validated when it loads. That would make the
contract above explicit and checked rather than documented.
