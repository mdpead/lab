"""Generation budget rules and batch independence.

Weights are random, so these assert the contract: the prompt is never modified,
max_new_tokens only reduces output, and a row does not depend on its batchmates.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from mlcore.generation import generate, generate_texts, make_kv_cache

DEV = "cuda"


def n_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def test_returns_one_text_and_one_flag_per_prompt(model, tokenizer):
    out, finished = generate_texts(
        model, tokenizer, ["hello", "the quick brown"], DEV, max_new_tokens=4
    )
    assert len(out) == 2
    assert len(finished) == 2
    assert all(isinstance(f, bool) for f in finished)


def test_respects_max_new_tokens(model, tokenizer):
    prompt = "hello"
    base = n_tokens(tokenizer, prompt)
    for new in (1, 5, 12):
        out, _ = generate_texts(model, tokenizer, [prompt], DEV, max_new_tokens=new)
        # eos may stop it early, so this is an upper bound
        assert n_tokens(tokenizer, out[0]) <= base + new


def test_output_is_independent_of_batchmates(model, tokenizer):
    """With an explicit budget, a row's result is the same batched as alone."""
    short, long = "hello", "the quick brown fox jumps over foo bar baz"
    alone = generate_texts(model, tokenizer, [short], DEV, max_new_tokens=15)[0][0]
    batched = generate_texts(model, tokenizer, [long, short], DEV, max_new_tokens=15)[0][1]
    assert alone == batched


def test_uncapped_budget_depends_on_batch(model, tokenizer):
    """With no cap, the longest prompt in the batch sets everyone's budget."""
    short, long = "hello", "the quick brown fox jumps over foo bar baz"
    alone = generate_texts(model, tokenizer, [short], DEV)[0][0]
    batched = generate_texts(model, tokenizer, [long, short], DEV)[0][1]
    assert n_tokens(tokenizer, alone) > n_tokens(tokenizer, batched)


def test_framing_tokens_are_stripped_from_the_text(model, tokenizer):
    """<bos>, padding and <eos> are framing, not model output.

    All three are forced into the stream, since random weights rarely emit <eos>.
    """
    short = f"hello {tokenizer.eos_token} world"
    long = "the quick brown fox jumps over foo bar baz"
    assert tokenizer.eos_token_id in tokenizer(short, add_special_tokens=False)["input_ids"]

    texts, _ = generate_texts(model, tokenizer, [long, short], DEV, max_new_tokens=15)

    for text in texts:
        for token in (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token):
            assert token not in text, f"{token} survived into {text!r}"


def test_other_special_tokens_survive_generation(model, tokenizer):
    """Reasoning markers survive: only the three framing tokens are dropped, by id."""
    marker = "<|think|>"
    assert len(tokenizer(marker, add_special_tokens=False)["input_ids"]) == 1

    texts, _ = generate_texts(model, tokenizer, [f"hello {marker} world"], DEV, max_new_tokens=3)
    assert marker in texts[0]


def test_finished_reports_eos_not_budget_exhaustion(model, tokenizer):
    """A one-token budget cannot reach <eos>, so nothing is reported as finished."""
    _, finished = generate_texts(model, tokenizer, ["hello"], DEV, max_new_tokens=1)
    assert finished == [False]


def test_prompt_is_never_truncated(model, tokenizer):
    """An over-long prompt is refused rather than silently trimmed."""
    too_long = "hello world foo bar baz qux " * 20
    assert n_tokens(tokenizer, too_long) > model.max_length
    with pytest.raises(ValueError, match="exceed"):
        generate_texts(model, tokenizer, [too_long], DEV)


def test_prompt_filling_context_generates_nothing(model, tokenizer):
    """Legal but has no room left; returns the prompt and says so."""
    filler = "hello world foo bar baz qux " * 20
    ids = tokenizer(filler, add_special_tokens=False)["input_ids"][: model.max_length - 1]
    exact = tokenizer.decode(ids)

    with pytest.warns(UserWarning, match="generating nothing"):
        out, _ = generate_texts(model, tokenizer, [exact], DEV)
    assert n_tokens(tokenizer, out[0]) == len(ids)


def test_max_new_tokens_clamped_with_warning(model, tokenizer):
    with pytest.warns(UserWarning, match="reduced"):
        generate_texts(model, tokenizer, ["hello"], DEV, max_new_tokens=10_000)


def test_training_mode_restored(model, tokenizer):
    model.train()
    try:
        generate_texts(model, tokenizer, ["hello"], DEV, max_new_tokens=2)
        assert model.training, "generate_texts left the model in eval mode"
    finally:
        model.eval()


def test_no_grad_leaks(model, tokenizer):
    """Generation must not build a graph or leave gradients behind."""
    for p in model.parameters():
        p.grad = None
    generate_texts(model, tokenizer, ["hello"], DEV, max_new_tokens=3)
    assert all(p.grad is None for p in model.parameters())


def test_cached_logits_match_uncached_forward(model, tokenizer):
    """Decoding with the kv cache gives the same logits, at every step, as a full
    forward over the same prefix."""
    prompts = ["hello", "the quick brown fox jumps over"]
    prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    prompt_ids = [[tokenizer.bos_token_id] + ids for ids in prompt_ids]
    encoded = tokenizer.pad({"input_ids": prompt_ids}, return_tensors="pt", padding_side="left")

    input_ids = encoded["input_ids"].to(DEV)
    batch_size, input_length = input_ids.shape
    seq_starts = (input_length - encoded["attention_mask"].sum(dim=-1)).to(
        device=DEV, dtype=torch.int32
    )

    new_tokens = 8
    token_ids = F.pad(input_ids, (0, new_tokens), value=tokenizer.pad_token_id)
    kv_cache = make_kv_cache(model, batch_size, input_length + new_tokens, DEV)

    num_cached = 0
    with torch.no_grad():
        for cur_length in range(input_length, input_length + new_tokens):
            cached, _, new_kv = model(
                token_ids[:, num_cached:cur_length], seq_starts, kv_cache[..., 0:num_cached, :]
            )
            kv_cache[..., num_cached:cur_length, :] = new_kv
            num_cached = cur_length

            full, _ = model(token_ids[:, 0:cur_length], seq_starts)

            a, b = cached[:, -1].float(), full[:, -1].float()
            err = (a - b).abs().max().item()
            scale = b.abs().max().item()
            step = cur_length - input_length
            assert err / scale < 1e-2, f"step {step}: cached logits diverge, rel {err / scale:.4g}"

            # Both paths continue from the reference's token, so a divergence is this step's
            token_ids[:, cur_length] = b.argmax(dim=-1)


def test_completion_mask_marks_only_generated_tokens(model, tokenizer):
    """True exactly where the model chose the token, so a row's True columns form one run."""
    prompts = ["hello", "the quick brown"]
    input_length = 1 + max(n_tokens(tokenizer, p) for p in prompts)  # +1 for <bos>

    token_ids, mask, _, _, _ = generate(model, tokenizer, prompts, DEV, max_new_tokens=6)

    assert mask.shape == token_ids.shape
    assert not mask[:, :input_length].any(), "prompt or left padding marked as generated"
    for row in mask:
        cols = row.nonzero().flatten()
        assert len(cols) > 0, "no generated token marked"
        expected = torch.arange(input_length, input_length + len(cols), device=cols.device)
        assert torch.equal(cols, expected), "masked columns are not one run from the frontier"


def force_eos_at(model, tokenizer, monkeypatch, row, step):
    """Make one row emit <eos> on a chosen step, which untrained weights never do unaided."""
    forward = model.forward
    calls = {"n": 0}

    def forced(*args, **kwargs):
        logits, aux, new_kv = forward(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == step:
            logits = logits.clone()
            logits[row] = -1e4
            logits[row, tokenizer.eos_token_id] = 1e4
        return logits, aux, new_kv

    monkeypatch.setattr(model, "forward", forced)


def test_completion_mask_covers_eos_but_not_the_filler_after_it(model, tokenizer, monkeypatch):
    """Stopping is a decision, so <eos> is masked in; the pad written afterwards is not."""
    stop_step = 2
    force_eos_at(model, tokenizer, monkeypatch, row=0, step=stop_step)

    token_ids, mask, _, _, finished = generate(
        model, tokenizer, ["hello", "the quick brown"], DEV, max_new_tokens=6
    )
    assert finished[0] and not finished[1], "the forced row should be the only one stopped"

    stopped, running = mask[0].nonzero().flatten(), mask[1].nonzero().flatten()
    assert len(stopped) == stop_step, "the stopped row masked in more than it generated"
    assert token_ids[0, stopped[-1]] == tokenizer.eos_token_id, "row 0 stopped elsewhere"
    assert not mask[0, stopped[-1] + 1 :].any(), "filler after <eos> marked as generated"
    assert running[-1] == mask.shape[1] - 1, "an unfinished row stopped short of its budget"


def test_seq_starts_points_at_the_first_real_token(model, tokenizer):
    """A caller rescoring these ids has to reproduce generation's padding to see its context."""
    token_ids, _, _, seq_starts, _ = generate(
        model, tokenizer, ["the quick brown fox jumps", "hello"], DEV, max_new_tokens=2
    )

    assert seq_starts[0].item() == 0, "the longest prompt should carry no padding"
    for ids, start in zip(token_ids, seq_starts):
        start = start.item()
        assert (ids[:start] == tokenizer.pad_token_id).all(), "real tokens before seq_start"
        assert ids[start] == tokenizer.bos_token_id, "seq_start does not land on <bos>"


def test_temperature_zero_is_deterministic(model, tokenizer):
    """Eval's contract: the same prompt gives the same answer, run to run."""
    prompts = ["hello", "the quick brown"]
    first = generate(model, tokenizer, prompts, DEV, max_new_tokens=10)[0]
    torch.manual_seed(1)  # a different RNG stream must not reach a greedy decode
    second = generate(model, tokenizer, prompts, DEV, max_new_tokens=10)[0]
    assert torch.equal(first, second)


def test_sampling_varies_with_the_seed(model, tokenizer):
    """And RL's: rollouts differ, or a group's advantages are all zero and nothing learns."""
    prompts = ["hello", "the quick brown"]
    first = generate(model, tokenizer, prompts, DEV, max_new_tokens=10, temperature=1.0)[0]
    torch.manual_seed(1)
    second = generate(model, tokenizer, prompts, DEV, max_new_tokens=10, temperature=1.0)[0]
    assert first.shape != second.shape or not torch.equal(first, second)


def test_sampling_never_draws_framing_tokens(model, tokenizer):
    """<bos>, <pad> and <unk> are never sampled; <eos> stays drawable."""
    token_ids, mask, _, _, _ = generate(
        model, tokenizer, ["hello", "the quick brown"], DEV, max_new_tokens=30, temperature=1.0
    )

    drawn = token_ids[mask]
    assert len(drawn) > 0
    for name in ("bos", "pad", "unk"):
        blocked = getattr(tokenizer, f"{name}_token_id")
        assert not (drawn == blocked).any(), f"sampled <{name}>, which decoding strips"


def test_negative_temperature_is_rejected(model, tokenizer):
    with pytest.raises(ValueError, match="temperature"):
        generate(model, tokenizer, ["hello"], DEV, max_new_tokens=2, temperature=-1.0)


def test_logprobs_only_come_back_from_a_sampled_decode(model, tokenizer):
    """Greedy returns None for logprobs rather than zeros a caller could train on."""
    greedy = generate(model, tokenizer, ["hello"], DEV, max_new_tokens=4)
    assert greedy[2] is None

    token_ids, _, logprobs, _, _ = generate(
        model, tokenizer, ["hello"], DEV, max_new_tokens=4, temperature=1.0
    )
    assert logprobs is not None
    assert logprobs.shape == token_ids.shape


def test_logprobs_score_the_token_actually_drawn(model, tokenizer):
    """At near-zero temperature each draw is near-certain, so its log-prob is near 0."""
    prompts = ["hello", "the quick brown"]
    sampled, mask, logprobs, _, _ = generate(
        model, tokenizer, prompts, DEV, max_new_tokens=6, temperature=0.01
    )
    greedy = generate(model, tokenizer, prompts, DEV, max_new_tokens=6)[0]

    assert torch.equal(sampled, greedy), "near-zero temperature diverged from greedy"
    assert (logprobs[mask] > math.log(0.5)).all(), "a near-certain draw scored below 0.5"
    assert (logprobs[mask] <= 0).all(), "a log-probability above log(1)"


def test_a_finished_row_leaves_no_infinite_logprob(model, tokenizer, monkeypatch):
    """A finished row's <pad> filler, masked to -inf, never reaches the logprobs."""
    force_eos_at(model, tokenizer, monkeypatch, row=0, step=2)

    _, mask, logprobs, _, finished = generate(
        model, tokenizer, ["hello", "the quick brown"], DEV, max_new_tokens=6, temperature=1.0
    )
    assert finished[0] and not finished[1], "the forced row should be the only one stopped"
    assert torch.isfinite(logprobs).all()
    assert torch.isfinite((logprobs * mask).sum())
