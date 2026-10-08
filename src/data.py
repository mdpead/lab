import hashlib
import itertools
import logging
import re

import datasets as hf_datasets
import numpy as np
from torch.utils.data import Dataset

from src import synth


# StarCoder prefixes ~48% of files with metadata markers on the first line, and the files
# do not parse as Python until they are gone. Anchored to the start of the file, and
# requiring the newline so a marker-only file is left alone.
_MARKER_PREFIX = re.compile(r"^<(?:reponame|filename|gh_stars)>[^\n]*\n")


def strip_markers(text):
    """Drop StarCoder's leading metadata line. Also applied when training the tokenizer."""
    return _MARKER_PREFIX.sub("", text, count=1)


def _hash_position(text):
    """Stable position in [0, 1) for a string. blake2b, because hash() is salted per process."""
    digest = hashlib.blake2b((text or "").encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


# The shard count is baked into the filenames, so naming a subset means hardcoding it
PYTHON_SHARDS = 59


def _shard_files(num_shards):
    if not 1 <= num_shards <= PYTHON_SHARDS:
        raise ValueError(f"num_shards must be 1..{PYTHON_SHARDS}, got {num_shards}")
    return [
        f"python/train-{i:05d}-of-{PYTHON_SHARDS:05d}.parquet" for i in range(num_shards)
    ]


def get_dataset_pretrain(config):
    """Raw StarCoder python files, split by repo. Rows carry a `content` column."""
    ds_config = config["data"]["pretrain"]

    num_shards = ds_config.get("num_shards")
    if num_shards is not None:
        # Download only the shards the step budget needs. Assumes the corpus was shuffled
        # before sharding, which shard 0 is consistent with but has not been checked.
        logging.info(f"pretrain: loading {num_shards} of {PYTHON_SHARDS} shards")
        ds = hf_datasets.load_dataset(
            "bigcode/starcoderdata", data_files=_shard_files(num_shards), split="train"
        )
    else:
        ds = hf_datasets.load_dataset(
            "bigcode/starcoderdata", data_dir="python", split="train", streaming=False
        )

    # Cap the row count for the toy configs
    sample_size = ds_config.get("sample_size")
    if sample_size is not None:
        ds = ds.select(range(min(sample_size, len(ds))))
        logging.info(f"pretrain: capped at {len(ds)} files")

    # Split on a hash of the repo, not on row index: rows are shuffled corpus-wide, so an
    # index cut puts files from the same repo on both sides and flatters validation loss.
    # The realised test fraction only approximates the configured ratio.
    ratio = ds_config["test_split_ratio"]

    def is_test(batch):
        return [_hash_position(name) < ratio for name in batch["max_stars_repo_name"]]

    def is_train(batch):
        return [not held_out for held_out in is_test(batch)]

    return {
        "test": ds.filter(is_test, batched=True, desc="Selecting test repos"),
        "train": ds.filter(is_train, batched=True, desc="Selecting train repos"),
    }


_TASK_FORMATTERS = {
    "output": synth.format_output_task,
    "input": synth.format_input_task,
    "search": synth.format_search_task,
}


# Parts of a task's completion that sft reads but is not scored on, as character spans.
_TASK_UNSCORED = {
    "search": synth.miss_argument_spans,
}


def _synth_records(ds_config, seed, with_trace):
    return synth.generate(
        ds_config["num_records"],
        tier=ds_config["tier"],
        seed=seed,
        inputs_per_fn=ds_config["inputs_per_fn"],
        with_trace=with_trace,
    )


def _split_on_source(rows, ratio):
    """Partition synth rows on the generated function, not the row.

    A function appears with several inputs and under several tasks, so a per-row split
    would leak. Plain lists: `args` mixes ints and lists, which arrow cannot type.
    """
    return {
        "test": [row for row in rows if _hash_position(row["source"]) < ratio],
        "train": [row for row in rows if _hash_position(row["source"]) >= ratio],
    }


def _is_test(source, ratio):
    """Which side of the split a generated function falls on. See _split_on_source."""
    return _hash_position(source) < ratio


def _sft_rows(config):
    """Yield prompt/completion rows from the synth generator, one per (record, task)."""
    ds_config = config["data"]["sft"]
    with_trace = ds_config["with_trace"]
    ratio = ds_config["test_split_ratio"]

    for record in _synth_records(ds_config, config["seed"], with_trace):
        is_test = _is_test(record["source"], ratio)
        for task in ds_config["tasks"]:
            prompt, completion = _TASK_FORMATTERS[task](record, with_trace)
            yield {
                "prompt": prompt,
                "completion": completion,
                "task": task,
                "source": record["source"],
                "n_executed": record["n_executed"],
                "is_test": is_test,
            }


def get_dataset_rl(config):
    """Prompts only, plus the fields synth's checkers need to score a rollout."""
    ds_config = config["data"]["rl"]

    # A seed of its own, so these are not the functions sft trained on
    rows = []
    for record in _synth_records(ds_config, config["seed"] + 1, with_trace=False):
        for task in ds_config["tasks"]:
            prompt, _ = _TASK_FORMATTERS[task](record, False)
            rows.append(
                {
                    "prompt": prompt,
                    "task": task,
                    "source": record["source"],
                    "args": record["args"],
                    "result": record["result"],
                    "n_executed": record["n_executed"],
                }
            )
    return _split_on_source(rows, ds_config["test_split_ratio"])


def get_dataset_sft(config):
    """Prompt/completion rows from the synth generator, one per (record, task).

    Rows carry `is_test` rather than being split here: prepare_sft splits after tokenising,
    which keeps the expensive traced generation to a single pass.
    """
    return hf_datasets.Dataset.from_generator(_sft_rows, gen_kwargs={"config": config})


def _tokenise_sft_batch(batch, tokenizer, max_length):
    """Map one batch of prompt/completion rows to model-ready sequences.

    Returning fewer rows than it was given is how an over-long task is dropped.
    """
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    prompts = tokenizer(batch["prompt"], add_special_tokens=False)["input_ids"]
    encoded = tokenizer(batch["completion"], add_special_tokens=False, return_offsets_mapping=True)
    completions = encoded["input_ids"]

    carried = ("task", "source", "n_executed", "is_test")
    out = {k: [] for k in ("ids", "prompt_len", "length", "unscored") + carried}
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        ids = [bos_id] + prompt + completion + [eos_id]
        # One over the context, for the input/target shift. Over-long tasks are dropped:
        # truncating would cut off the answer.
        if len(ids) > max_length + 1:
            continue
        out["ids"].append(ids)
        out["prompt_len"].append(1 + len(prompt))
        out["length"].append(len(ids) - 1)
        # Positions in `ids` of tokens touching an unscored span. A typed array, so a batch
        # with none still writes an integer column.
        spans = _TASK_UNSCORED.get(batch["task"][i], lambda completion: [])(batch["completion"][i])
        out["unscored"].append(
            np.array(
                [
                    1 + len(prompt) + j
                    for j, (start, end) in enumerate(encoded["offset_mapping"][i])
                    if any(start < hi and end > lo for lo, hi in spans)
                ],
                dtype=np.int32,
            )
        )
        for key in carried:
            out[key].append(batch[key][i])
    return out


def prepare_sft(ds_raw, tokenizer, config):
    """Tokenise prompt/completion pairs into single sequences with a mask boundary.

    `prompt_len` is where the completion starts in `ids`; targets before it are masked.
    `length` is the model input length, which the sampler batches on. Rows stay
    memory-mapped and randomly indexable.
    """
    max_length = config["model"]["max_length"]

    prepared = ds_raw.map(
        _tokenise_sft_batch,
        batched=True,
        batch_size=1000,
        remove_columns=ds_raw.column_names,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        desc="Tokenizing sft",
    )
    # map() can serve a cached result from before `unscored` existed, and the collate
    # would treat the missing column as nothing to mask.
    if "unscored" not in prepared.column_names:
        raise RuntimeError(
            f"sft: stale tokenised cache without `unscored`: delete {prepared.cache_files}"
        )
    dropped = len(ds_raw) - len(prepared)
    if dropped:
        logging.warning(
            f"sft: dropped {dropped}/{len(ds_raw)} rows over max_length {max_length}"
        )

    # Split here rather than at generation, so nothing has to be buffered
    return {
        "test": prepared.filter(lambda b: b["is_test"], batched=True),
        "train": prepared.filter(lambda b: [not v for v in b["is_test"]], batched=True),
    }


def prepare_rl(ds_raw, tokenizer, config):
    """Annotate rollout prompts with their token length, for length-bucketed batching.

    The prompt stays text, and the reward fields ride along untouched.
    """
    max_length = config["model"]["max_length"]
    prepared = {}
    for split, rows in ds_raw.items():
        encoded = tokenizer([row["prompt"] for row in rows], add_special_tokens=False)

        kept, dropped = [], 0
        for row, prompt in zip(rows, encoded["input_ids"]):
            length = 1 + len(prompt)  # +1 for the <bos> generate_texts prepends
            # A prompt that fills the context leaves nothing to generate
            if length >= max_length:
                dropped += 1
                continue
            kept.append({**row, "length": length})
        if dropped:
            logging.warning(
                f"rl {split}: dropped {dropped}/{len(rows)} prompts with no room to generate"
            )
        prepared[split] = kept
    return prepared


def _tokenize_and_chunk_batch(batch, tokenizer, max_length):
    """Tokenise, join with <eos> separators, and cut fixed blocks in one pass.

    Blocks are packed across file boundaries, so nothing is spent on padding. The remainder
    of each map batch, under one block, is dropped.
    """
    # +1 token per block, for the input/target shift in collate
    block = max_length + 1
    eos_id = tokenizer.eos_token_id
    cleaned = [strip_markers(text) for text in batch["content"]]
    encoded = tokenizer(cleaned, add_special_tokens=False)["input_ids"]

    concatenated = list(itertools.chain.from_iterable(ids + [eos_id] for ids in encoded))
    total = (len(concatenated) // block) * block
    return {"ids": [concatenated[i : i + block] for i in range(0, total, block)]}


class TokenizedDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]["ids"]


def _check_token_budget(n_blocks, config):
    """Report whether the loaded shards cover the configured step budget.

    Undersizing does not raise: the sampler loops, so training quietly repeats data.
    """
    train_config = config["train"]["pretrain"]
    needed = (
        train_config["num_steps"]
        * train_config["effective_batch_token_size"]
        // config["model"]["max_length"]
    )
    if n_blocks < needed:
        logging.warning(
            f"pretrain: {n_blocks} blocks available but {needed} needed for "
            f"{train_config['num_steps']} steps — training will repeat data "
            f"({needed / n_blocks:.2f} epochs). Raise data.pretrain.num_shards."
        )
    else:
        logging.info(
            f"pretrain: {n_blocks} blocks available, {needed} needed "
            f"({100 * needed / n_blocks:.0f}% of one epoch)"
        )


def prepare_pretrain(ds_raw, tokenizer, config):
    max_length = config["model"]["max_length"]

    # int32 rather than the int64 arrow infers: halves storage for a 32000-token vocabulary
    features = hf_datasets.Features({"ids": hf_datasets.Sequence(hf_datasets.Value("int32"))})

    prepared = {}
    for split, ds in ds_raw.items():
        chunked = ds.map(
            _tokenize_and_chunk_batch,
            batched=True,
            remove_columns=ds.column_names,
            features=features,
            fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
            desc=f"Tokenizing and chunking {split}",
        )
        prepared[split] = TokenizedDataset(chunked.with_format("torch"))

    _check_token_budget(len(prepared["train"]), config)
    return prepared
