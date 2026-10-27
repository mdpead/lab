"""SFT data preparation: streaming into arrow, the split, and the chunk boundary."""

import datasets
import pytest

from src import data, dataloader


@pytest.fixture(autouse=True)
def _no_arrow_cache():
    """Disable the datasets cache, which would otherwise mask changes to the mapper."""
    datasets.disable_caching()
    yield
    datasets.enable_caching()


def _config(tokenizer, num_records, max_length=512, ratio=0.1, tasks=("output",)):
    return {
        "seed": 0,
        "model": {"max_length": max_length},
        "data": {
            "sft": {
                "num_records": num_records,
                "tier": "easy",
                "tasks": list(tasks),
                "inputs_per_fn": 4,
                "with_trace": True,
                "test_split_ratio": ratio,
            }
        },
    }


def _prepared(tokenizer, num_records, **kw):
    cfg = _config(tokenizer, num_records, **kw)
    return cfg, data.prepare_sft(data.get_dataset_sft(cfg), tokenizer, cfg)


def test_prepared_rows_are_memory_mapped_not_resident(tokenizer):
    """Rows live on disk and are addressed by index, not held as a list."""
    _, ds = _prepared(tokenizer, 200)

    for split in ("train", "test"):
        assert ds[split].cache_files, f"{split} is not backed by an arrow file"
        assert not isinstance(ds[split], list)


def test_rows_stay_randomly_indexable(tokenizer):
    """Length bucketing needs random access."""
    _, ds = _prepared(tokenizer, 200)
    train = ds["train"]

    assert train[0]["length"] == train[0]["length"]
    assert len(train["length"]) == len(train)
    assert train[len(train) - 1]["ids"]


def test_no_function_appears_in_both_splits(tokenizer):
    """No generated function appears on both sides of the split."""
    _, ds = _prepared(tokenizer, 300)

    train_sources = set(ds["train"]["source"])
    test_sources = set(ds["test"]["source"])
    assert train_sources and test_sources
    assert not (train_sources & test_sources)


def test_tokenising_keeps_every_row_that_fits(tokenizer):
    """The batched map neither drops a row that fits nor duplicates one."""
    cfg, ds = _prepared(tokenizer, 120)

    assert len(ds["train"]) + len(ds["test"]) == len(data.get_dataset_sft(cfg))


def test_token_sampler_batches_arrow_and_lists_identically(tokenizer):
    """The sampler packs the same batches from an arrow column as from a list of rows."""
    _, ds = _prepared(tokenizer, 200)
    train = ds["train"]
    as_list = [dict(row) for row in train]

    from_arrow = dataloader.TokenSampler(train, 2048, seed=0).batches
    from_list = dataloader.TokenSampler(as_list, 2048, seed=0).batches

    assert from_arrow == from_list


def test_rows_longer_than_the_context_are_dropped(tokenizer):
    """prepare_sft drops rather than truncates, since truncating cuts off the answer."""
    cfg = _config(tokenizer, 200)
    raw = data.get_dataset_sft(cfg)
    loose = data.prepare_sft(raw, tokenizer, cfg)
    # ds[col] is an arrow Column, not a list — it does not concatenate with +.
    lengths = sorted(list(loose["train"]["length"]) + list(loose["test"]["length"]))
    # A bound inside the observed spread, so some rows fit and some do not
    bound = lengths[len(lengths) // 2]

    tight_cfg = _config(tokenizer, 200, max_length=bound)
    tight = data.prepare_sft(data.get_dataset_sft(tight_cfg), tokenizer, tight_cfg)

    kept_tight = len(tight["train"]) + len(tight["test"])
    assert 0 < kept_tight < len(lengths)
    assert all(n <= bound for n in tight["train"]["length"])


def test_rl_prompts_are_not_the_sft_functions(tokenizer):
    """RL scores answers, so functions sft showed the answers to would reward recall."""
    cfg = _config(tokenizer, 400)
    cfg["data"]["rl"] = {
        "num_records": 200,
        "tier": "easy",
        "tasks": ["output"],
        "inputs_per_fn": 4,
        "test_split_ratio": 0.1,
    }

    def rl_sources():
        return {row["source"] for rows in data.get_dataset_rl(cfg).values() for row in rows}

    sft_sources = {r["source"] for r in data._synth_records(cfg["data"]["sft"], cfg["seed"], True)}

    assert rl_sources(), "rl generated nothing"
    # Chance collisions between the two streams are allowed; sharing a stream is not
    assert len(rl_sources() & sft_sources) < 0.05 * len(rl_sources())
    assert rl_sources() == rl_sources(), "rl data is not deterministic"


def _lossless_tokenizer():
    """One token per byte, so targets can be read back as text and no token straddles a span."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        [""],
        trainers.BpeTrainer(
            vocab_size=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[
                "<bos>",
                "<eos>",
                "<pad>",
                "<|think|>",
                "<|/think|>",
                "<|answer|>",
                "<|/answer|>",
            ],
        ),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>"
    )


def test_search_rows_leave_exactly_the_missed_guesses_unscored():
    """The collated targets, read back as text, are the completion minus each miss's arguments."""
    from src import synth

    tokenizer = _lossless_tokenizer()
    cfg, ds = _prepared(tokenizer, 60, max_length=2048, tasks=("search",))
    pad = tokenizer.pad_token_id
    rows = [ds["train"][i] for i in range(len(ds["train"]))]
    assert any(len(row["unscored"]) for row in rows) and any(
        not len(row["unscored"]) for row in rows
    )

    for row in rows:
        targets = dataloader.collate_sft_batch([row], pad)["output_ids"][0]
        completion = tokenizer.decode(row["ids"][row["prompt_len"] : -1])
        expected = completion
        for lo, hi in reversed(synth.miss_argument_spans(completion)):
            expected = expected[:lo] + expected[hi:]
        scored = tokenizer.decode([t for t in targets.tolist()[:-1] if t != pad])
        assert scored == expected


def test_tasks_with_nothing_unscored_still_write_the_column(tokenizer):
    """An output-only dataset still writes a typed, empty `unscored` column."""
    _, ds = _prepared(tokenizer, 20)
    assert all(len(row["unscored"]) == 0 for row in ds["train"])
    assert "int" in str(ds["train"].features["unscored"])
