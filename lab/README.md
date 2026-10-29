# The `lab` package

The library the projects in this repo build on: model layers, Triton kernels, generation, typed
configs, and the training infrastructure around a project's own loss. This page explains how
it is organised and why: where the line between library and project falls, how a project
plugs in, and how each change was checked to leave results unchanged.

## Who uses what

| | `en_cy_translation` | `python_reasoning` |
|---|---|---|
| `models` | `encoder_decoder`, built by `build_model` | `decoder_only`, built by `build_model` |
| `layers`, `kernels` | through its model | through its model, including Triton flash attention |
| `config` | its schema, built on `RunConfig`, `StageConfig` and `SupervisedStage` | its schema, built on the same, plus `TrainStage` for RL |
| `training` | `train`, `supervised_loop` | `train`, `supervised_loop` for pretraining and SFT; `train` with its own RL loop |
| `data.samplers` | `TokenSampler` | `InfiniteRandomSampler`, `TokenSampler` |
| `generation` | | KV-cache decoding, for evaluation and RL rollouts |
| `runs`, `tokenizer` | yes | yes |

## Where the line falls

**The library owns mechanics; the project owns meaning.** The library runs the parts that are
the same whatever the task: gradient accumulation, mixed precision, clipping, the validation
and checkpoint schedule, exact resume, run directories, and checking configs. The project
decides what a batch contains, what the loss is, what counts as a good result, and how its
data is built and what text its tokenizer learns from.

The config follows the same line. The package declares the settings it reads, and a project's
schema subclasses those classes to add its own, so every key in a config file belongs to
exactly one class.

**Code moves in when a second project needs it, not before.** Nothing was designed in advance.
The package began as the parts of `python_reasoning` that `en_cy_translation` duplicated, and each
later move followed the same rule:

| Stayed in the project | Why |
|---|---|
| The RL loop (`python_reasoning/src/rl.py`) | One user, and it does not fit the supervised shape: it generates rollouts, scores them by running code, and trains on group-relative advantages. It still uses `training.train` for resume and checkpoints. |
| Encoder-decoder decoding (`en_cy_translation/src/generation.py`, greedy, for BLEU during training; beam search comes from its Hugging Face wrapper) | One user. `lab.generation` is the decoder-only KV-cache path the reasoning project needs. |
| Batch collation, data preparation, the text a tokenizer trains on | Task-specific by nature: they decide what a batch means. |
| BLEU, execution-checked answers | The task's definition of success, passed to the library as a callback. |
| Data and task settings in the config | Only the project reads them, so only its schema declares them. |

**Changes are checked to change nothing.** Each move was verified to leave results unchanged
before any redesign, and interface changes are kept small and stated; see
[How each change was checked](#how-each-change-was-checked).

## Extension points

A project plugs in with plain functions, decorated classes and dataclasses rather than by
subclassing the training code:

| Hook | What the project supplies | Used by |
|---|---|---|
| `compute_loss` | `(model, batch) -> (loss, values_to_log)` | `training.supervised_loop` |
| `tokens_key` | the batch field whose non-pad tokens count towards throughput | `training.supervised_loop` |
| `validation_metrics` | `(model, batch) -> dict[str, float]`, optional | `training.supervised_loop` |
| `loop` | `(stage, model, dataloaders, tokenizer, run, config) -> None` | `training.train` |
| `corpus` | `(datasets) -> Iterable[str]`, the text the tokenizer trains on | `tokenizer.get_tokenizer` |
| A component | an `nn.Module` registered with `@NORMS.register("name")` (or `FEEDFORWARDS`, `ATTENTIONS`, `POSITIONS`); its constructor is its config | `model.*` blocks with that `type` |
| A tokenizer | a class registered with `@TOKENIZERS.register("name")` whose `train(texts)` returns a `PreTrainedTokenizerFast`; its constructor is its config | the `tokenizer` section with that `type` |
| A schema | frozen dataclasses subclassing `RunConfig`, `StageConfig` and `TrainStage` or `SupervisedStage` | `runs.parse_config`, `runs.init_run` |

The translation project's whole training module is a loss, a BLEU callback and two calls into
`lab`:

```python
def compute_loss(model, batch, criterion):
    logits = model(
        batch["src_input_ids"],
        batch["tgt_input_ids"],
        batch["src_padding_mask"],
        batch["tgt_padding_mask"],
    )
    loss = criterion(logits.reshape(-1, logits.shape[2]), batch["tgt_output_ids"].reshape(-1))
    return loss, {"loss": loss}


def train_loop(stage, model, dataloaders, tokenizer, run, config):
    training.supervised_loop(
        stage,
        model,
        dataloaders,
        tokenizer,
        run,
        config,
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

A component is a layer and a decorator. Its config block is checked against its constructor,
leaving out the arguments the model passes in (`d_model` here):

```python
NORMS = Registry("norm", provided=("d_model",))


@NORMS.register("rms")
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None: ...
```

`norm: {type: rms, eps: 1.0e-6}` then builds `RMSNorm(d_model, eps=1e-6)`. A layer can define
`check_config(**settings)` for checks across settings, such as grouped-query attention's
"`num_heads` must be a multiple of `num_kv_heads`", so they fail when the config loads.

## The config

Each project has YAML files in `configs/`, grouped by stage, with model components named:

```yaml
name: base
seed: 0
model:
  type: decoder_only
  d_model: 1024
  num_layers: 16
  max_length: 1024
  dropout: 0.0
  norm: {type: rms}
  position: {type: rope}
  attention: {type: gqa, num_heads: 16, num_kv_heads: 4, d_h: 64}
  feedforward: {type: moe, d_ff: 1408, num_experts: 4, top_k: 2, capacity_factor: 1.25}
stages:
  sft:
    init_from: {stage: pretrain, step: 10000}
    train: {learning_rate: 3.0e-4, ...}
    data: {tier: easy, tasks: [output], ...}
tokenizer: {type: bpe, vocab_size: 32000, training_size: 50000, digit_split: true, ...}
locations: {models_dir: ./models}
```

Every script checks its config against the project's schema before anything runs, and the
first problem fails with its file and path:

```
configs/sft_only.yaml: stages.sft.train: unknown key 'num_step'; did you mean 'num_steps'?
configs/sft_only.yaml: stages.sft.train.learning_rate: expected a number, got str '3e-4' (YAML needs a dot in exponent floats, e.g. 3.0e-4)
configs/base.yaml: model.feedforward: feedforward type 'moe' is not supported here (one of: relu, swiglu)
```

| Part | Declared by |
|---|---|
| `name`, `seed`, `locations` | `runs.RunConfig` |
| `model` | the family's `ModelConfig` (`decoder_only`, `encoder_decoder`), chosen by `type`; each component block by its layer's constructor |
| `stages.<stage>.init_from` | `training.InitFrom`: the earlier stage, and optionally the checkpoint step, a stage's weights start from |
| `stages.<stage>.train` | `training.TrainStage` (any stage) or `SupervisedStage` (a stage `supervised_loop` runs), with the project's own fields added |
| `tokenizer` | the tokenizer's constructor, chosen by `type`: `wordpiece` or `bpe` (`lab.tokenizer.TOKENIZERS`) |
| `stages.<stage>.data` | the project's schema |

**Which changes restart a stage** is declared on the setting. Fields marked `resumable()`, such
as `num_steps`, `checkpoint_steps` and `compile_model`, can change and the stage resumes;
anything else in the model, tokenizer, the stage's own section, the name or the seed restarts
it. Configs are compared after parsing, so writing out a default is the same as leaving it out.

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

1. **`runs.init_run(config, stage, schema)`** creates the directories and compares the stage's
   fingerprint with the saved config's. If a setting the checkpoints depend on changed, the
   stage's directory is cleared, so a stale checkpoint is never resumed under a new setting.
   If the saved config cannot be read against the schema at all, the run stops instead: the
   checkpoints may be worth keeping. A changed tokenizer section clears the saved tokenizer.
2. **`training.train(...)`** builds the run state. It resumes from the stage's latest
   checkpoint if one exists. Otherwise it starts fresh, seeding the weights from the stage
   named in `init_from` if set, with a fresh optimiser and step counter. If `num_steps` is
   already reached it returns.
3. **The loop** fast-forwards the sampler to the exact minibatch it stopped at, without reading
   any data, then trains. It checkpoints every `checkpoint_steps` and once more at the end, and
   keeps the newest `keep_checkpoints`.

## How each change was checked

| Change | Check |
|---|---|
| Extract the shared code from `python_reasoning` ([#2](https://github.com/mdpead/lab/pull/2)) | Every line of the original files compared against the new layout; the same 243 tests before and after, with the same 101 passing on CPU. |
| Add the encoder-decoder and move `en_cy_translation` onto the package and newer dependencies ([#3](https://github.com/mdpead/lab/pull/3)) | Weight names kept, so existing checkpoints and the published model load. The published model's greedy and beam-search translations identical across old and new code and versions, with logits within 1e-5. |
| Share the supervised loop ([#5](https://github.com/mdpead/lab/pull/5)) | Bitwise-identical `results.json`, apart from timings, before and after: translation on CPU and GPU, reasoning SFT on GPU. |
| Share the test fixtures ([#7](https://github.com/mdpead/lab/pull/7)) | The same tests collected and passing in all three suites before and after. |
| Check configs against typed schemas ([#10](https://github.com/mdpead/lab/pull/10)) | Every config file parses; the schema's resumable fields give the same fingerprint as the old hard-coded key list for every stage of every config. |
| Build models from named components, stage-first configs ([#11](https://github.com/mdpead/lab/pull/11)) | Against fixtures recorded from the old code: identical seeded weights for both families, identical parameter names, shapes and counts for every config, and bitwise-identical encoder-decoder outputs. Each rewritten config holds the same values. A one-off script migrates saved runs; on run folders recreated in the old layout, every stage resumed with its checkpoint. |
| Share the tokenizers ([#14](https://github.com/mdpead/lab/pull/14)) | Against fixtures recorded from each project's tokenizer code: an identical serialized BPE tokenizer, with and without digit splitting. WordPiece training breaks ties randomly, so its learned vocabulary differs between runs of the old code too; its normalizer, pre-tokenizer, decoder, post-processor and special-token ids are identical. The migration script brings saved configs to the new section, so trained tokenizers are kept. |

## Testing

The package's tests are in `tests/` at the repo root. Tests that build a decoder-only model or call
a Triton kernel are marked `cuda` and skip without a GPU; CI runs the rest on every change.
`lab.testing` is a pytest plugin with the shared markers and fixtures, loaded from a
conftest with `pytest_plugins = ["lab.testing"]`. `tests/fixtures/` and each project's
`tests/fixtures/` hold the model layouts and weight hashes recorded from the code before #11,
which the model tests compare against, and the tokenizers recorded before #14.

## Limitations and next steps

- **Code reads the dicts, not the parsed config.** Configs are parsed to check them and to
  compare fingerprints, but the code then reads the YAML's dicts by key. Moving it to the
  typed objects would let editors and type checkers follow settings through the code.
- **Some training choices are fixed.** The optimiser is AdamW with weight decay 0.01, and
  gradients are clipped at norm 1.0. The next step is to make them configurable, with their
  defaults matching today's values.
- **`runs.parse_config` reads `configs/<name>.yaml`** relative to the working directory, so
  scripts run from the project folder.
- **`supervised_loop` takes the tokenizer only for its pad id.**
- **Generation is decoder-only.** The encoder-decoder's decoding stays in its project until a
  second encoder-decoder needs it.
- **Two projects is a small sample.** The boundaries fit both, but a third project would test
  whether they generalise.
