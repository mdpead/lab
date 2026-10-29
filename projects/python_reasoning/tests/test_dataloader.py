"""Data loading for both trained stages: the input/target shift, the samplers, the masks.

Built on synthetic rows. The SFT half pins two things that fail quietly: the
prompt/completion mask boundary, and that output_ids is a copy rather than a view.
"""

import itertools
import json
from pathlib import Path

import pytest
import torch
from lab.data.samplers import InfiniteRandomSampler, TokenSampler, resume_at_minibatch
from lab.testing import batch_digest

from src.dataloader import (
    PromptSampler,
    collate_lm_batch,
    collate_rl_batch,
    collate_sft_batch,
    create_dataloaders_pretrain,
    create_dataloaders_rl,
    create_dataloaders_sft,
)

MAX_LENGTH = 16
BLOCK = MAX_LENGTH + 1  # prepare_pretrain stores one extra token for the shift
MINIBATCH_TOKENS = 64
NUM_BLOCKS = 20
PROMPTS_PER_STEP = 4
NUM_PROMPTS = 14  # deliberately not a multiple of PROMPTS_PER_STEP


class _Blocks(torch.utils.data.Dataset):
    """Stands in for data.TokenizedDataset: one fixed-length int32 block per item."""

    def __init__(self, n, pad_id, block=BLOCK):
        # Distinct, non-pad ids, so any mis-shift shows up as a value mismatch
        self.rows = [
            torch.arange(r * block, (r + 1) * block, dtype=torch.int32) + pad_id + 1
            for r in range(n)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


@pytest.fixture
def config():
    return {
        "seed": 0,
        "model": {"max_length": MAX_LENGTH},
        "stages": {
            "pretrain": {
                "train": {
                    "minibatch_token_size": MINIBATCH_TOKENS,
                    "num_workers": 2,
                    "prefetch_factor": 2,
                }
            },
            "rl": {"train": {"generation_prompts_size": PROMPTS_PER_STEP}},
        },
    }


@pytest.fixture
def splits(tokenizer):
    return {
        "train": _Blocks(NUM_BLOCKS, tokenizer.pad_token_id),
        "test": _Blocks(NUM_BLOCKS // 2, tokenizer.pad_token_id),
    }


# --- collate ---


def test_collate_shifts_inputs_and_targets_by_one(tokenizer):
    rows = _Blocks(3, tokenizer.pad_token_id)
    batch = collate_lm_batch([rows[i] for i in range(3)], tokenizer.pad_token_id)

    assert batch["input_ids"].shape == (3, MAX_LENGTH)
    assert batch["output_ids"].shape == (3, MAX_LENGTH)
    assert batch["padding_mask"].shape == (3, MAX_LENGTH)

    for i in range(3):
        # Target at position t is the input at position t+1
        assert torch.equal(batch["output_ids"][i, :-1], batch["input_ids"][i, 1:].long())


def test_collate_output_ids_are_long_for_cross_entropy(tokenizer):
    """The ids are stored int32; CrossEntropyLoss requires Long targets."""
    rows = _Blocks(2, tokenizer.pad_token_id)
    assert rows[0].dtype == torch.int32

    batch = collate_lm_batch([rows[0], rows[1]], tokenizer.pad_token_id)
    assert batch["output_ids"].dtype == torch.int64


def test_collate_padding_mask_marks_exactly_the_pad_positions(tokenizer):
    pad = tokenizer.pad_token_id
    row = torch.arange(BLOCK, dtype=torch.int32) + pad + 1
    row[-4:] = pad  # a tail of padding, as a short final block would have

    batch = collate_lm_batch([row], pad)

    # The mask is taken before the shift, so it lines up with input_ids.
    assert torch.equal(batch["padding_mask"][0], batch["input_ids"][0] != pad)
    assert batch["padding_mask"][0, :-3].all()
    assert not batch["padding_mask"][0, -3:].any()


# --- dataloader construction ---


def test_batch_size_is_the_token_budget_divided_by_context(splits, tokenizer, config):
    dls = create_dataloaders_pretrain(splits, tokenizer, config)
    for split, dl in dls.items():
        assert dl.batch_size == MINIBATCH_TOKENS // MAX_LENGTH, split


def test_train_split_samples_infinitely_and_test_split_does_not(splits, tokenizer, config):
    dls = create_dataloaders_pretrain(splits, tokenizer, config)
    assert isinstance(dls["train"].sampler, InfiniteRandomSampler)
    assert not isinstance(dls["test"].sampler, InfiniteRandomSampler)


def test_test_split_runs_in_process(splits, tokenizer, config):
    """Validation is a handful of batches; paying worker startup for it would dominate."""
    dls = create_dataloaders_pretrain(splits, tokenizer, config)
    assert dls["train"].num_workers == 2
    assert dls["train"].prefetch_factor == 2
    assert dls["test"].num_workers == 0
    assert dls["test"].prefetch_factor is None


def test_prefetch_factor_is_dropped_when_workers_are_disabled(splits, tokenizer, config):
    """DataLoader rejects a prefetch_factor alongside num_workers=0."""
    config["stages"]["pretrain"]["train"]["num_workers"] = 0
    dls = create_dataloaders_pretrain(splits, tokenizer, config)
    for dl in dls.values():
        assert dl.num_workers == 0
        assert dl.prefetch_factor is None


def test_batches_have_the_shape_the_model_expects(splits, tokenizer, config):
    config["stages"]["pretrain"]["train"]["num_workers"] = 0  # keep the test in-process
    dls = create_dataloaders_pretrain(splits, tokenizer, config)
    batch_size = MINIBATCH_TOKENS // MAX_LENGTH

    for i, batch in enumerate(dls["train"]):
        assert set(batch) == {"input_ids", "padding_mask", "output_ids"}
        assert batch["input_ids"].shape == (batch_size, MAX_LENGTH)
        assert batch["output_ids"].shape == (batch_size, MAX_LENGTH)
        assert batch["padding_mask"].shape == (batch_size, MAX_LENGTH)
        if i == 2:
            break
    else:
        pytest.fail("train loader ended; InfiniteRandomSampler should never exhaust it")


def test_test_loader_covers_the_split_once(splits, tokenizer, config):
    config["stages"]["pretrain"]["train"]["num_workers"] = 0
    dls = create_dataloaders_pretrain(splits, tokenizer, config)
    batch_size = MINIBATCH_TOKENS // MAX_LENGTH

    batches = list(dls["test"])
    seen = sum(batch["input_ids"].shape[0] for batch in batches)

    # drop_last defaults to False, so validation sees the whole split
    assert seen == len(splits["test"])
    assert len(batches) == -(-len(splits["test"]) // batch_size)


# --- sft rows ---


def _sft_row(n, prompt_len, pad_id, tag=0):
    """A prepare_sft row: <bos> + prompt + completion + <eos>, flattened to `n` ids.

    Ids are distinct and never equal pad_id, so a mis-shift or stray mask shows up.
    """
    return {
        "ids": [tag * 1000 + i + pad_id + 1 for i in range(n)],
        "prompt_len": prompt_len,
        "length": n - 1,
    }


# --- sft collate ---


def test_sft_collate_masks_the_prompt_and_scores_the_completion(tokenizer):
    """The shift puts the first completion target at prompt_len - 1, and no prompt is scored."""
    pad = tokenizer.pad_token_id
    row = _sft_row(n=7, prompt_len=4, pad_id=pad)
    out = collate_sft_batch([row], pad)["output_ids"][0]

    assert (out[: row["prompt_len"] - 1] == pad).all()
    assert out[row["prompt_len"] - 1 :].tolist() == row["ids"][row["prompt_len"] :]


def test_sft_collate_leaves_the_prompt_intact_in_the_inputs(tokenizer):
    """output_ids must be a copy, not a view, or masking it would blank the prompt input."""
    pad = tokenizer.pad_token_id
    row = _sft_row(n=7, prompt_len=4, pad_id=pad)
    batch = collate_sft_batch([row], pad)

    assert batch["input_ids"][0].tolist() == row["ids"][:-1]
    assert pad not in batch["input_ids"][0].tolist()


def test_sft_collate_scores_the_stop_token(tokenizer):
    """<eos> is the last target: nothing else teaches the model to stop generating."""
    pad = tokenizer.pad_token_id
    row = _sft_row(n=7, prompt_len=4, pad_id=pad)
    out = collate_sft_batch([row], pad)["output_ids"][0]

    assert out[-1].item() == row["ids"][-1]


def test_sft_collate_pads_ragged_rows_to_the_batch_maximum(tokenizer):
    pad = tokenizer.pad_token_id
    rows = [
        _sft_row(n=9, prompt_len=3, pad_id=pad, tag=0),
        _sft_row(n=5, prompt_len=2, pad_id=pad, tag=1),
    ]
    batch = collate_sft_batch(rows, pad)

    # One narrower than the longest row, since the shift consumes a position.
    assert batch["input_ids"].shape == (2, 8)
    assert batch["output_ids"].shape == (2, 8)

    # The long row fills the width; the short one is real up to its own length, then pad.
    assert batch["padding_mask"][0].all()
    assert batch["padding_mask"][1, :5].all()
    assert not batch["padding_mask"][1, 5:].any()


def test_sft_collate_masks_padding_and_prompt_with_the_same_id(tokenizer):
    """criterion is built with a single ignore_index, so both must be pad_token_id."""
    pad = tokenizer.pad_token_id
    rows = [
        _sft_row(n=9, prompt_len=3, pad_id=pad, tag=0),
        _sft_row(n=5, prompt_len=2, pad_id=pad, tag=1),
    ]
    out = collate_sft_batch(rows, pad)["output_ids"]

    scored = out != pad
    # Row 0: 9 ids -> 8 targets, 2 of them prompt. Row 1: 5 ids -> 4 targets, 1 prompt,
    # then 4 positions of padding out to the batch width.
    assert scored[0].sum().item() == 8 - (3 - 1)
    assert scored[1].sum().item() == 4 - (2 - 1)


def test_sft_collate_targets_are_long_for_cross_entropy(tokenizer):
    pad = tokenizer.pad_token_id
    batch = collate_sft_batch([_sft_row(n=6, prompt_len=3, pad_id=pad)], pad)
    assert batch["output_ids"].dtype == torch.int64


def test_sft_collate_returns_the_keys_the_supervised_loop_reads(tokenizer):
    """Same keys as collate_lm_batch, so the supervised loop works the same for both stages."""
    pad = tokenizer.pad_token_id
    batch = collate_sft_batch([_sft_row(n=6, prompt_len=3, pad_id=pad)], pad)
    assert set(batch) == {"input_ids", "padding_mask", "output_ids"}


# --- sft dataloader construction ---


@pytest.fixture
def sft_config():
    return {
        "seed": 0,
        "model": {"max_length": MAX_LENGTH},
        "stages": {
            "sft": {
                "train": {
                    "minibatch_token_size": MINIBATCH_TOKENS,
                    "num_workers": 2,
                    "prefetch_factor": 2,
                }
            }
        },
    }


@pytest.fixture
def sft_splits(tokenizer):
    pad = tokenizer.pad_token_id
    return {
        "train": [_sft_row(n=6 + (t % 5), prompt_len=3, pad_id=pad, tag=t) for t in range(12)],
        "test": [_sft_row(n=6 + (t % 3), prompt_len=2, pad_id=pad, tag=t) for t in range(5)],
    }


def test_sft_test_split_runs_in_process(sft_splits, tokenizer, sft_config):
    dls = create_dataloaders_sft(sft_splits, tokenizer, sft_config)
    assert dls["train"].num_workers == 2
    assert dls["test"].num_workers == 0
    assert dls["test"].prefetch_factor is None


def test_sft_loader_batches_by_token_budget_not_row_count(sft_splits, tokenizer, sft_config):
    """batch_size stays None: the batch_sampler owns the grouping."""
    dls = create_dataloaders_sft(sft_splits, tokenizer, sft_config)
    for dl in dls.values():
        assert dl.batch_size is None
        assert isinstance(dl.batch_sampler, TokenSampler)


def test_sft_loader_reports_no_length(sft_splits, tokenizer, sft_config):
    """The SFT loader is endless, so it refuses a length."""
    dls = create_dataloaders_sft(sft_splits, tokenizer, sft_config)
    for split, dl in dls.items():
        with pytest.raises(TypeError):
            len(dl)


def test_sft_batches_have_the_shape_the_model_expects(sft_splits, tokenizer, sft_config):
    sft_config["stages"]["sft"]["train"]["num_workers"] = 0  # keep the test in-process
    dls = create_dataloaders_sft(sft_splits, tokenizer, sft_config)
    pad = tokenizer.pad_token_id

    for i, batch in enumerate(dls["train"]):
        assert set(batch) == {"input_ids", "padding_mask", "output_ids"}
        rows, width = batch["input_ids"].shape
        assert batch["output_ids"].shape == (rows, width)
        assert batch["padding_mask"].shape == (rows, width)
        # Rows vary in length, so unlike pretrain the width is per batch, not the context.
        assert width <= MAX_LENGTH
        assert (batch["output_ids"] != pad).any()
        if i == 2:
            break
    else:
        pytest.fail("train loader ended; TokenSampler should never exhaust it")


# --- resume ---


def test_resume_converts_minibatches_to_rows_for_pretrain(splits, tokenizer, config):
    """The pretrain sampler indexes rows and the loop counts minibatches; batch_size converts."""
    dls = create_dataloaders_pretrain(splits, tokenizer, config)
    batch_size = MINIBATCH_TOKENS // MAX_LENGTH

    resume_at_minibatch(dls["train"], 5)

    assert dls["train"].sampler.start_index == 5 * batch_size
    assert dls["train"].sampler.start_minibatch == 5


def test_resume_needs_no_conversion_for_sft(sft_splits, tokenizer, sft_config):
    """TokenSampler yields whole minibatches, so the loop's unit is already its own."""
    dls = create_dataloaders_sft(sft_splits, tokenizer, sft_config)

    resume_at_minibatch(dls["train"], 5)

    assert dls["train"].batch_sampler.start_minibatch == 5


def test_resume_finds_the_sampler_that_actually_drives_iteration(sft_splits, tokenizer, sft_config):
    """Resume state goes on the batch_sampler, not the throwaway .sampler DataLoader builds."""
    dls = create_dataloaders_sft(sft_splits, tokenizer, sft_config)
    assert not isinstance(dls["train"].sampler, TokenSampler)

    resume_at_minibatch(dls["train"], 4)

    assert dls["train"].batch_sampler.start_minibatch == 4
    assert not hasattr(dls["train"].sampler, "start_minibatch")


def test_resume_rejects_a_sampler_with_no_resume_state(splits, tokenizer, config):
    """A stock sampler with no resume state is rejected rather than silently accepted."""
    dls = create_dataloaders_pretrain(splits, tokenizer, config)
    with pytest.raises(TypeError, match="start_minibatch"):
        resume_at_minibatch(dls["test"], 5)


# --- rl ---


@pytest.fixture
def rl_rows():
    """Stands in for prepare_rl's output. Lengths are shuffled relative to index, so a
    sampler that ignored `length` would produce visibly mixed batches."""
    lengths = [40, 12, 33, 7, 25, 19, 48, 3, 30, 15, 44, 9, 21, 37][:NUM_PROMPTS]
    return {
        "train": [
            {
                "prompt": f"def f(x): return x + {i}\n\n# What does f(1) return?\n",
                "length": length,
                "task": "output",
                "source": f"def f(x): return x + {i}",
                "args": [1],
                "result": 1 + i,
            }
            for i, length in enumerate(lengths)
        ],
        "test": [
            {
                "prompt": f"def g(x): return x * {i}\n\n# What does g(2) return?\n",
                "length": 10 + i,
                "task": "output",
                "source": f"def g(x): return x * {i}",
                "args": [2],
                "result": 2 * i,
            }
            for i in range(NUM_PROMPTS // 2)
        ],
    }


def test_rl_collate_passes_the_scoring_fields_through(rl_rows):
    """The reward comes from executing the record, so the row has to survive the collate."""
    batch = collate_rl_batch(rl_rows["train"][:3])
    assert len(batch) == 3
    for original, collated in zip(rl_rows["train"][:3], batch):
        assert collated == original
    assert all(isinstance(row["prompt"], str) for row in batch), "prompts were tokenised"


def test_prompt_sampler_fills_every_batch_but_the_last(rl_rows):
    sampler = PromptSampler(rl_rows["train"], PROMPTS_PER_STEP, seed=0)
    sizes = [len(b) for b in sampler.batches]
    assert sizes[:-1] == [PROMPTS_PER_STEP] * (len(sizes) - 1)
    assert 0 < sizes[-1] <= PROMPTS_PER_STEP
    assert sum(sizes) == NUM_PROMPTS, "the short final batch was dropped"


def test_prompt_sampler_groups_similar_lengths(rl_rows):
    """Batches hold prompts of similar length, so little generation goes on padding."""
    rows = rl_rows["train"]
    sampler = PromptSampler(rows, PROMPTS_PER_STEP, seed=0)

    spans = [
        max(rows[i]["length"] for i in batch) - min(rows[i]["length"] for i in batch)
        for batch in sampler.batches
    ]
    whole_split = max(r["length"] for r in rows) - min(r["length"] for r in rows)
    assert max(spans) < whole_split / 2, f"batches span nearly the whole range: {spans}"


def test_prompt_sampler_covers_every_prompt_once_per_pass(rl_rows):
    sampler = PromptSampler(rl_rows["train"], PROMPTS_PER_STEP, seed=0)
    seen = [i for batch in itertools.islice(iter(sampler), len(sampler.batches)) for i in batch]
    assert sorted(seen) == list(range(NUM_PROMPTS))


def test_prompt_sampler_resumes_where_it_left_off(rl_rows):
    """A resumed run continues the shuffle rather than replaying the pass."""
    full = list(itertools.islice(iter(PromptSampler(rl_rows["train"], PROMPTS_PER_STEP, 0)), 6))
    resumed = PromptSampler(rl_rows["train"], PROMPTS_PER_STEP, 0, start_minibatch=2)
    assert list(itertools.islice(iter(resumed), 4)) == full[2:]


def test_rl_train_loader_is_infinite_and_test_loader_is_one_pass(rl_rows, tokenizer, config):
    loaders = create_dataloaders_rl(rl_rows, tokenizer, config)

    passes = -(-NUM_PROMPTS // PROMPTS_PER_STEP)
    taken = list(itertools.islice(loaders["train"], passes * 3))
    assert len(taken) == passes * 3, "the train loader ran out"

    test_batches = list(loaders["test"])
    seen = [row["prompt"] for batch in test_batches for row in batch]
    assert len(seen) == len(rl_rows["test"]), "the test loader skipped or repeated prompts"
    assert seen == [row["prompt"] for row in rl_rows["test"]], "test order is not the split's"


def test_rl_loaders_run_in_process(rl_rows, tokenizer, config):
    """Workers exist to overlap tensor building with the GPU; this collate builds none."""
    loaders = create_dataloaders_rl(rl_rows, tokenizer, config)
    for loader in loaders.values():
        assert loader.num_workers == 0


def test_rl_train_loader_resumes(rl_rows, tokenizer, config):
    """resume_at_minibatch finds the batch_sampler rather than the unused plain sampler."""
    loaders = create_dataloaders_rl(rl_rows, tokenizer, config)
    resume_at_minibatch(loaders["train"], 2)
    assert loaders["train"].batch_sampler.start_minibatch == 2


# --- batches recorded from the loaders before they moved into lab ---

REFERENCE = json.loads((Path(__file__).parent / "fixtures/batches.json").read_text())


def test_loaders_yield_the_recorded_batches(splits, sft_splits, tokenizer, config, sft_config):
    config["stages"]["pretrain"]["train"]["num_workers"] = 0
    sft_config["stages"]["sft"]["train"]["num_workers"] = 0
    pretrain = create_dataloaders_pretrain(splits, tokenizer, config)
    sft = create_dataloaders_sft(sft_splits, tokenizer, sft_config)
    digests = {
        "pretrain_train": batch_digest(pretrain["train"], 30),
        "pretrain_test": batch_digest(pretrain["test"], 30),
        "sft_train": batch_digest(sft["train"], 30),
        "sft_test": batch_digest(sft["test"], 30),
    }
    assert digests == REFERENCE
