"""Synthetic execution-reasoning task generation."""

import pytest

from src import synth


@pytest.mark.parametrize("inputs_per_fn", [0, 1])
def test_generate_rejects_too_few_inputs(inputs_per_fn):
    """Fewer than two inputs can never satisfy the distinctness filter, and would hang."""
    with pytest.raises(ValueError, match="inputs_per_fn must be at least 2"):
        list(synth.generate(1, tier="easy", inputs_per_fn=inputs_per_fn))


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_generate_produces_solvable_records(tier):
    """Every record must carry a result the checker accepts, for each difficulty tier."""
    records = list(synth.generate(6, tier=tier, seed=0, inputs_per_fn=4))
    assert len(records) == 6
    for r in records:
        _, completion = synth.format_output_task(r, include_trace=True)
        assert synth.check_output_answer(r, completion), f"{tier}: own answer rejected"
        assert r["tier"] == tier


def test_generated_functions_depend_on_their_arguments():
    """The filter exists so output prediction cannot be memorised from the source alone."""
    records = list(synth.generate(24, tier="easy", seed=3, inputs_per_fn=4))
    by_source = {}
    for r in records:
        by_source.setdefault(r["source"], set()).add(repr(r["result"]))
    assert any(len(v) > 1 for v in by_source.values()), "no function varied with its input"


# --- completion format ---


def _record():
    return list(synth.generate(1, tier="easy", seed=42, inputs_per_fn=4))[0]


def test_completion_wraps_trace_and_answer_in_markers():
    """The markers are the contract between the formatter and the reward path."""
    r = _record()
    _, completion = synth.format_output_task(r, include_trace=True)

    assert completion.startswith(synth.THINK_OPEN)
    assert completion.endswith(synth.ANSWER_CLOSE)
    trace = completion.split(synth.THINK_OPEN)[1].split(synth.THINK_CLOSE)[0]
    answer = completion.split(synth.ANSWER_OPEN)[1].split(synth.ANSWER_CLOSE)[0]
    assert trace == r["trace"]
    assert answer == repr(r["result"])


def test_untraced_completion_has_no_thinking_markers():
    """RL generates untraced, so the prompt must not imply a trace the reward ignores."""
    r = _record()
    _, completion = synth.format_output_task(r, include_trace=False)

    assert synth.THINK_OPEN not in completion
    assert completion == f"{synth.ANSWER_OPEN}{r['result']!r}{synth.ANSWER_CLOSE}"


def test_markers_cannot_occur_in_generated_source():
    """The delimiters must never appear in generated source, or the answer would split wrongly."""
    for r in synth.generate(200, tier="easy", seed=11, inputs_per_fn=4):
        assert "<|" not in r["source"]
        assert "<|" not in r["trace"]


# --- extract_answer ---


def test_extract_answer_reads_a_complete_rollout():
    r = _record()
    _, completion = synth.format_output_task(r, include_trace=True)
    assert synth.extract_answer(completion) == repr(r["result"])


def test_extract_answer_recovers_a_rollout_truncated_before_the_closing_marker():
    """A rollout cut off mid-answer still has a usable first line."""
    r = _record()
    _, completion = synth.format_output_task(r, include_trace=True)
    truncated = completion.replace(synth.ANSWER_CLOSE, "")

    assert synth.extract_answer(truncated) == repr(r["result"])


def test_extract_answer_stops_at_the_first_line_when_generation_runs_on():
    r = _record()
    _, completion = synth.format_output_task(r, include_trace=True)
    runaway = completion.replace(synth.ANSWER_CLOSE, "\nand then some more text")

    assert synth.extract_answer(runaway) == repr(r["result"])


def test_extract_answer_is_none_when_no_answer_was_reached():
    """Distinct from a wrong answer: the checker must score it False rather than raise."""
    assert synth.extract_answer(f"{synth.THINK_OPEN}a = 4") is None
    assert synth.extract_answer("") is None


# --- search task ---


def _search_records(n=200):
    return list(synth.generate(n, tier="easy", seed=7, inputs_per_fn=4))


def _split_search(completion):
    """A formatted search completion as (attempts, answer)."""
    return synth.extract_attempts(completion), synth.extract_answer(completion)


def test_search_completion_ends_on_the_first_attempt_that_reaches_the_target():
    """Every attempt before the last misses and the last hits, so stopping tracks the result."""
    counts = set()
    for r in _search_records():
        prompt, completion = synth.format_search_task(r, include_trace=True)
        attempts, answer = _split_search(completion)
        returned = [a.split("\n")[-1].split("  ->  ")[-1] for a in attempts]

        assert 1 <= len(attempts) <= synth.MAX_MISSES + 1
        assert all(value != repr(r["result"]) for value in returned[:-1])
        assert returned[-1] == repr(r["result"])
        assert attempts[-1].split("\n")[0] == f"{synth.TRY_PREFIX}{answer}"
        assert synth.check_input_answer(r, prompt + completion)
        counts.add(len(attempts))

    assert counts == set(range(1, synth.MAX_MISSES + 2)), "every attempt count should occur"


def test_search_completion_is_deterministic_and_asks_the_input_question():
    """Misses are seeded from the record, and the prompt is the input task's unchanged."""
    r = _search_records(1)[0]
    assert synth.format_search_task(r, True) == synth.format_search_task(r, True)
    assert synth.format_search_task(r, False) == synth.format_input_task(r, False)


def test_every_attempt_sft_shows_is_a_faithful_one():
    """The formatter and the reward's faithfulness check agree on what an attempt looks like."""
    for r in _search_records():
        prompt, completion = synth.format_search_task(r, include_trace=True)
        attempts = synth.extract_attempts(completion)
        expected = 1.0 + synth.REWARD_FAITHFUL - synth.ATTEMPT_COST * (len(attempts) - 1)
        assert synth.reward_search_answer(r, prompt + completion) == pytest.approx(expected)


_SEARCH_RECORD = {
    "source": "def f(a, b):\n    a += b\n    b = (a * 2)\n    return a, b",
    "result": (5, 10),
}


def _search_rollout(think, answer):
    return (
        f"{_SEARCH_RECORD['source']}\n\n# What arguments make f return (5, 10)?\n"
        f"{synth.THINK_OPEN}{think}{synth.THINK_CLOSE}{synth.ANSWER_OPEN}{answer}{synth.ANSWER_CLOSE}"
    )


_HIT = "try 2, 3\na += b  ->  a = 5\nb = (a * 2)  ->  b = 10\nreturn a, b  ->  (5, 10)"
_MISS = "try 1, 3\na += b  ->  a = 4\nb = (a * 2)  ->  b = 8\nreturn a, b  ->  (4, 8)"


def test_search_reward_ranks_rollouts_the_answer_alone_leaves_tied():
    """The graded terms order rollouts that a correct-or-not reward scores the same."""
    reward = lambda think, answer: synth.reward_search_answer(
        _SEARCH_RECORD, _search_rollout(think, answer)
    )

    # Right answer: an honest trace beats an invented one, and one attempt beats two
    invented = _HIT.replace("a = 5", "a = 6")
    assert reward(_HIT, "2, 3") > reward(_MISS + "\n" + _HIT, "2, 3") > reward(invented, "2, 3")
    assert reward(invented, "2, 3") == pytest.approx(1.0)

    # Wrong answer: an honest miss beats claiming the miss reached the target
    claimed = _MISS.replace("(4, 8)", "(5, 10)")
    assert reward(_MISS, "1, 3") > reward(claimed, "1, 3") == 0.0

    # An answer that does not come from the last attempt loses that attempt's credit
    assert reward(_HIT, "2, 3") > reward(_HIT, "5, 0") == pytest.approx(1.0)

    # Any right answer outranks any wrong one, however the thinking went
    assert reward(invented + "\n" + invented, "2, 3") > reward(_MISS, "1, 3")


def test_search_reward_credits_each_returned_value_a_wrong_answer_gets():
    record = {"source": "def f(a, b):\n    b = (b + 1)\n    return a, b", "result": (4, 7)}
    rollout = lambda answer: f"{synth.ANSWER_OPEN}{answer}{synth.ANSWER_CLOSE}"

    assert synth.reward_search_answer(record, rollout("4, 0")) == synth.REWARD_MATCH / 2
    assert synth.reward_search_answer(record, rollout("0, 0")) == 0.0
    assert synth.reward_search_answer(record, rollout("4, 6")) == 1.0


def test_closest_credit_grades_a_wrong_answer_by_how_near_it_lands():
    record = {"source": "def f(a, b):\n    b = (b + 1)\n    return a, b", "result": (4, 7)}
    reward = lambda answer: synth.reward_search_answer(
        record, f"{synth.ANSWER_OPEN}{answer}{synth.ANSWER_CLOSE}", partial_credit="closest"
    )

    # Each step towards the target is worth more, with no value yet exactly right
    assert 0 < reward("40, 60") < reward("9, 20") < reward("5, 9") < reward("4, 5")
    # However near, a miss stays under REWARD_MATCH and a hit is worth the full 1.0
    assert reward("4, 5") < synth.REWARD_MATCH < reward("4, 6") == 1.0


def test_closest_credit_never_falls_for_searching_on():
    """Graded on the best guess, not the last, so a later guess that lands further away
    costs only its attempt. Graded on the last, stopping after one guess scored better."""
    far = "try 9, 9\na += b  ->  a = 18\nb = (a * 2)  ->  b = 36\nreturn a, b  ->  (18, 36)"
    reward = lambda think, answer: synth.reward_search_answer(
        _SEARCH_RECORD, _search_rollout(think, answer), partial_credit="closest"
    )

    stopped, searched_on = reward(_MISS, "1, 3"), reward(_MISS + "\n" + far, "9, 9")
    assert searched_on == pytest.approx(stopped - synth.ATTEMPT_COST)
    # Counted from running the guess, so a trace that misreports it changes nothing here
    lied = far.replace("(18, 36)", "(5, 10)")
    assert reward(_MISS + "\n" + lied, "9, 9") < searched_on


def test_search_reward_refuses_an_unknown_partial_credit():
    with pytest.raises(ValueError, match="partial_credit"):
        synth.reward_search_answer(_SEARCH_RECORD, "", partial_credit="nearest")


def test_closeness_is_one_only_for_the_target_itself():
    assert synth._closeness((4, [1, 2]), (4, [1, 2])) == 1.0
    assert synth._closeness(7, 7) == 1.0
    assert synth._closeness((4, [1, 3]), (4, [1, 2])) < 1.0
    # A list of the wrong length loses the elements it is missing or adds
    assert synth._closeness([1, 2], [1, 2, 3]) == pytest.approx(2 / 3)
    assert synth._closeness([1, 2, 3, 4], [1, 2]) == pytest.approx(1 / 2)
    # Nothing comparable: a different type, a different arity, no result at all
    assert synth._closeness("4", 4) == synth._closeness((4,), (4, 7)) == 0.0
    assert synth._closeness(synth._NO_RESULT, (4, 7)) == 0.0


@pytest.mark.parametrize(
    "completion",
    [
        "",
        f"{synth.THINK_OPEN}try 1, 3\na += b",  # out of budget mid-attempt
        f"{synth.THINK_OPEN}try 1,\n{synth.THINK_CLOSE}{synth.ANSWER_OPEN}1,{synth.ANSWER_CLOSE}",
        f"{synth.THINK_OPEN}try nonsense{synth.THINK_CLOSE}"
        f"{synth.ANSWER_OPEN}(({synth.ANSWER_CLOSE}",
        f"{synth.THINK_OPEN}try 'x', []{synth.THINK_CLOSE}"
        f"{synth.ANSWER_OPEN}'x', []{synth.ANSWER_CLOSE}",
    ],
)
def test_search_reward_scores_malformed_rollouts_without_raising(completion):
    """A sampled rollout can be anything, and one that raised would take the step with it."""
    assert synth.reward_search_answer(_SEARCH_RECORD, completion) <= 0.0


def _stats(think, answer, *keys):
    stats = synth.search_stats(_SEARCH_RECORD, _search_rollout(think, answer))
    return {key: stats[key] for key in keys}


def test_search_stats_separate_a_first_try_hit_from_a_later_one():
    keys = ("attempts", "first_try_solved", "faithful", "solved_later")

    assert _stats(_HIT, "2, 3", *keys) == {
        "attempts": 1,
        "first_try_solved": 1.0,
        "faithful": 1.0,
        "solved_later": 0.0,
    }
    assert _stats(_MISS + "\n" + _HIT, "2, 3", *keys) == {
        "attempts": 2,
        "first_try_solved": 0.0,
        "faithful": 1.0,
        "solved_later": 1.0,
    }
    # Decided by running the first guess, not by what the rollout says it returned
    claimed = _MISS.replace("(4, 8)", "(5, 10)")
    assert _stats(claimed, "1, 3", *keys) == {
        "attempts": 1,
        "first_try_solved": 0.0,
        "faithful": 0.0,
        "solved_later": 0.0,
    }
    assert set(synth.search_stats(_SEARCH_RECORD, "").values()) == {0.0}


def test_search_stats_count_the_attempts_that_get_nearer():
    """An attempt improves when it lands nearer the target than every one before it."""
    far = "try 9, 9\na += b  ->  a = 18\nb = (a * 2)  ->  b = 36\nreturn a, b  ->  (18, 36)"
    keys = ("later_attempts", "improving_attempts")

    assert _stats(_HIT, "2, 3", *keys) == {"later_attempts": 0, "improving_attempts": 0}
    assert _stats(far + "\n" + _MISS + "\n" + _HIT, "2, 3", *keys) == {
        "later_attempts": 2,
        "improving_attempts": 2,
    }
    # Getting nearer than the last attempt is not enough, and a repeat is not nearer
    assert _stats(_MISS + "\n" + far + "\n" + _MISS, "1, 3", *keys) == {
        "later_attempts": 2,
        "improving_attempts": 0,
    }
    # Counted from running the guess: a trace that claims the target gains nothing
    claimed = far.replace("(18, 36)", "(5, 10)")
    assert _stats(_MISS + "\n" + claimed, "9, 9", *keys)["improving_attempts"] == 0


def test_blind_guessing_is_expected_to_improve_once_in_k_on_the_kth_attempt():
    three = _MISS + "\n" + _MISS + "\n" + _MISS
    assert _stats(_MISS, "1, 3", "improving_if_blind") == {"improving_if_blind": 0}
    assert _stats(three, "1, 3", "improving_if_blind")["improving_if_blind"] == pytest.approx(
        1 / 2 + 1 / 3
    )


def test_miss_argument_spans_cover_every_guess_but_the_one_that_hits():
    """SFT leaves unscored the arguments of each miss, and nothing else."""
    seen_misses = False
    for r in _search_records():
        _, completion = synth.format_search_task(r, include_trace=True)
        attempts = synth.extract_attempts(completion)
        spanned = [completion[lo:hi] for lo, hi in synth.miss_argument_spans(completion)]

        assert spanned == [a.split("\n")[0][len(synth.TRY_PREFIX) :] for a in attempts[:-1]]
        seen_misses |= bool(spanned)
    assert seen_misses

    _, untraced = synth.format_search_task(_search_records(1)[0], include_trace=False)
    assert synth.miss_argument_spans(untraced) == []
