"""The RL loop moves the policy toward rewarded rollouts.

The real reward needs a model that can already answer, so these swap in one a random model
hits about half the time: whether its one generated token starts with a space.
"""

import pytest
import torch
import torch.nn.functional as F

from src import rl, train
from src.dataloader import create_dataloaders_rl

DEV = "cuda"
PROMPTS = ["hello", "the quick", "foo bar", "brown fox"]


def _continuation(row, completion):
    # decode_rollouts returns the prompt and its continuation as one string
    assert completion.startswith(row["prompt"])
    return completion[len(row["prompt"]):]


def _rewarded_ids(tokenizer):
    special = set(tokenizer.all_special_ids)
    return torch.tensor(
        [
            i
            for i in range(len(tokenizer))
            if i not in special and tokenizer.decode([i]).startswith(" ")
        ],
        device=DEV,
    )


def _mass_on(model, tokenizer, ids):
    """Mean probability, over PROMPTS, that the first generated token is one of `ids`."""
    masses = []
    with torch.no_grad():
        for prompt in PROMPTS:
            x = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            x = torch.tensor([[tokenizer.bos_token_id] + x], device=DEV)
            logits, _ = model(x)
            masses.append(F.softmax(logits[0, -1].float(), dim=-1)[ids].sum().item())
    return sum(masses) / len(masses)


def _run(
    model, tokenizer, tmp_path, monkeypatch, checker, num_steps, temperature=1.0, max_new_tokens=1,
    reward=None, **overrides,
):
    monkeypatch.setitem(rl._TASK_CHECKERS, "fake", checker)
    if reward is not None:
        monkeypatch.setitem(rl._TASK_REWARDS, "fake", reward)

    train_config = {
        "device": DEV,
        "learning_rate": 1e-3,
        "adam_betas": [0.9, 0.95],
        "adam_eps": 1e-8,
        "warm_up_steps": 1,
        "generation_prompts_size": 2,
        "effective_prompts_size": 4,
        "completions_per_prompt": 16,
        "num_steps": num_steps,
        "checkpoint_steps": 1000,
        "validation_steps": 1000,
        "validation_batches": 1,
        "router_aux_loss_coef": 0.01,
        # One token by default, the position _mass_on reads
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        **overrides,
    }
    config = {"seed": 0, "model": {"max_length": 64}, "train": {"rl": train_config}}

    rows = [
        {"prompt": p, "task": "fake", "length": 1 + len(tokenizer(p, add_special_tokens=False)["input_ids"])}
        for p in PROMPTS
    ]
    dataloaders = create_dataloaders_rl({"train": rows, "test": rows}, tokenizer, config)

    _, optimiser, lr_scheduler, scaler = train.create_training_objects(model, train_config, tokenizer)
    run = {
        "optimiser": optimiser,
        "lr_scheduler": lr_scheduler,
        "scaler": scaler,
        "results": [],
        "step_no": 0,
        "run_path": str(tmp_path),
    }
    rl.train_loop("rl", model, dataloaders, tokenizer, run, config)
    return run["results"]


def _space_led(row, completion):
    """The fake reward: a random model hits it often enough that most groups are mixed."""
    return _continuation(row, completion).startswith(" ")


def _train(results):
    return [r for r in results if r["type"] == "train"]


@pytest.fixture
def fresh_model(make_model, tokenizer):
    # fp32 so the optimiser step is not lost to fp16 rounding at this scale
    return make_model(len(tokenizer)).float()


@pytest.mark.parametrize("sign", [1, -1], ids=["rewarded", "penalised"])
def test_update_moves_mass_toward_reward(fresh_model, tokenizer, tmp_path, monkeypatch, sign):
    """Rewarding space-led tokens raises their probability; rewarding the rest lowers it."""
    ids = _rewarded_ids(tokenizer)
    before = _mass_on(fresh_model, tokenizer, ids)
    assert 0.1 < before < 0.9, "the fake reward needs mixed groups to produce a gradient"

    def checker(row, completion):
        hit = _continuation(row, completion).startswith(" ")
        return hit if sign > 0 else not hit

    results = _run(fresh_model, tokenizer, tmp_path, monkeypatch, checker, num_steps=20)
    after = _mass_on(fresh_model, tokenizer, ids)

    assert sign * (after - before) > 0.05, f"mass went {before:.3f} -> {after:.3f}"
    assert all(r["mixed_group_rate"] > 0 for r in results if r["type"] == "train")


def test_uniform_groups_are_dropped_and_leave_the_weights_alone(
    fresh_model, tokenizer, tmp_path, monkeypatch
):
    """With every group agreeing nothing is rescored; the step counts but nothing is applied."""
    before = [p.detach().clone() for p in fresh_model.parameters()]
    results = _run(fresh_model, tokenizer, tmp_path, monkeypatch, lambda row, c: True, num_steps=2)
    train_results = _train(results)

    assert [r["step_no"] for r in train_results] == [1, 2]
    assert all(r["trained_groups"] == 0 for r in train_results)
    assert all(r["loss_pg"] == 0 and r["grad_norm"] == 0 for r in train_results)
    assert all(r["mixed_group_rate"] == 0 for r in train_results)
    assert all(r["reward"] == 1 for r in train_results), "reward is over every group, dropped or not"
    assert all(torch.equal(a, b) for a, b in zip(before, fresh_model.parameters()))


def test_only_mixed_groups_are_trained_on(fresh_model, tokenizer, tmp_path, monkeypatch):
    """Groups for one prompt always agree; only the others reach the rescore."""
    def checker(row, completion):
        return True if row["prompt"] == "hello" else _space_led(row, completion)

    results = _train(_run(fresh_model, tokenizer, tmp_path, monkeypatch, checker, num_steps=3))

    # 4 prompts a step, one of them uniform by construction
    assert all(r["trained_groups"] <= 3 for r in results)
    assert any(r["trained_groups"] > 0 for r in results)
    assert all(r["mixed_group_rate"] == pytest.approx(r["trained_groups"] / 4) for r in results)


def test_a_graded_reward_trains_while_the_checker_still_reports_success(
    fresh_model, tokenizer, tmp_path, monkeypatch
):
    """The gradient follows a task's graded reward; its checker is what is reported."""
    results = _run(
        fresh_model, tokenizer, tmp_path, monkeypatch, lambda row, c: False, num_steps=2,
        reward=lambda row, c: 0.5 * _space_led(row, c),
    )
    train_results = _train(results)

    assert all(r["trained_groups"] > 0 and r["grad_norm"] > 0 for r in train_results)
    assert all(0 < r["reward"] < 0.5 for r in train_results)
    assert all(r["solved_rate"] == 0 for r in train_results)
    assert all(r["pass_rate"] == 0 for r in results if r["type"] == "validation")


def test_advantage_sd_floor_shrinks_the_update_from_a_narrow_group(
    fresh_model, tokenizer, tmp_path, monkeypatch
):
    """Without a floor a group split by 0.01 is normalised up to a full-size advantage.

    Same rollouts and seed, so the gradients differ only by the floor. The bound is loose
    because the router's aux gradient does not scale with the advantage.
    """
    narrow = lambda row, c: 0.01 * _space_led(row, c)
    state = {k: v.clone() for k, v in fresh_model.state_dict().items()}

    norms = []
    for overrides in ({}, {"advantage_sd_floor": 0.5}):
        fresh_model.load_state_dict(state)
        torch.manual_seed(0)
        results = _run(
            fresh_model, tokenizer, tmp_path, monkeypatch, _space_led, num_steps=1,
            reward=narrow, **overrides,
        )
        norms.append(_train(results)[0]["grad_norm"])

    assert 0 < norms[1] < norms[0] / 10


def test_a_fresh_run_records_its_starting_point(fresh_model, tokenizer, tmp_path, monkeypatch):
    results = _run(fresh_model, tokenizer, tmp_path, monkeypatch, _space_led, num_steps=1)

    assert results[0]["type"] == "validation" and results[0]["step_no"] == 0
    assert "sampled_reward" in results[0]


def test_rescore_matches_sampling_logprobs(fresh_model, tokenizer, tmp_path, monkeypatch):
    """The gradient pass scores the same tokens the sampler drew, to within precision."""
    results = _train(_run(fresh_model, tokenizer, tmp_path, monkeypatch, _space_led, num_steps=1))
    assert results[0]["trained_groups"] > 0, "nothing was rescored, so the gap is vacuous"
    assert results[0]["logprob_gap"] < 1e-3


def test_rescore_matches_sampling_logprobs_at_other_temperatures(
    fresh_model, tokenizer, tmp_path, monkeypatch
):
    """The gradient pass scores the tempered distribution the sampler drew from."""
    results = _train(
        _run(fresh_model, tokenizer, tmp_path, monkeypatch, _space_led, num_steps=1, temperature=2.0)
    )
    assert results[0]["trained_groups"] > 0, "nothing was rescored, so the gap is vacuous"
    assert results[0]["logprob_gap"] < 1e-3


def test_sampled_validation_is_reproducible_and_leaves_training_rng(fresh_model, tokenizer, monkeypatch):
    """Same weights, same score, and training's own random stream is not advanced."""
    monkeypatch.setitem(rl._TASK_CHECKERS, "fake", lambda row, c: _continuation(row, c).startswith(" "))
    rows = [
        {"prompt": p, "task": "fake", "length": 1 + len(tokenizer(p, add_special_tokens=False)["input_ids"])}
        for p in PROMPTS
    ]
    config = {"seed": 0, "train": {"rl": {"generation_prompts_size": 2}}}
    loader = create_dataloaders_rl({"test": rows}, tokenizer, config)["test"]

    def validate():
        return rl.validation_step(fresh_model, loader, tokenizer, torch.device(DEV), 0, 1, 10, 16, 1.0, 0)

    torch.manual_seed(123)
    expected_next = torch.rand(1, device=DEV)
    torch.manual_seed(123)
    first = validate()
    assert torch.equal(torch.rand(1, device=DEV), expected_next), "validation consumed the global RNG"

    assert validate() == first
    assert 0 < first["sampled_reward"] < 1
    assert first["num_prompts"] == len(PROMPTS)


def test_rescore_lines_up_over_multi_token_rollouts(fresh_model, tokenizer, tmp_path, monkeypatch):
    """Each rescored token is matched to the one the sampler drew, across prompts of
    different lengths."""
    results = _train(
        _run(fresh_model, tokenizer, tmp_path, monkeypatch, _space_led, num_steps=1, max_new_tokens=12)
    )
    assert results[0]["num_completion_tokens"] > 16 * 4 * 2, "rollouts were not multi-token"
    assert results[0]["trained_groups"] > 0, "nothing was rescored, so the gap is vacuous"
    assert results[0]["logprob_gap"] < 1e-3


def test_token_budget_chunks_fit_the_budget_and_cover_every_row():
    starts = [0, 0, 4, 4, 2, 2]
    ends = [10, 30, 12, 9, 20, 11]

    chunks = rl.token_budget_chunks(starts, ends, budget=40)

    assert sorted(i for chunk in chunks for i in chunk) == list(range(6))
    for chunk in chunks:
        width = max(ends[i] for i in chunk) - min(starts[i] for i in chunk)
        assert len(chunk) == 1 or len(chunk) * width <= 40
    # Short rows are grouped together rather than padded out to the longest
    assert [1] in chunks


def test_token_budget_chunks_keeps_an_oversized_row_and_defaults_to_one_chunk():
    assert rl.token_budget_chunks([0, 0], [100, 5], budget=10) == [[1], [0]]
    assert rl.token_budget_chunks([0, 0, 0], [7, 3, 5], budget=None) == [[1, 2, 0]]


def test_chunked_rescore_gives_the_single_pass_gradient(fresh_model, tokenizer):
    """Splitting the rescore to fit a token budget changes peak memory, not the update."""
    from src import generation

    prompts = [p for p in ["hello", "the quick brown fox jumps over", "foo bar baz"] for _ in range(4)]
    torch.manual_seed(0)
    token_ids, completion_mask, old_logprobs, seq_starts, _ = generation.generate(
        fresh_model, tokenizer, prompts, DEV, max_new_tokens=12, temperature=1.0
    )
    advantages = torch.randn(len(prompts), device=DEV)
    blocked = generation.blocked_token_ids(tokenizer, torch.device(DEV))
    fresh_model.train()

    def grads(budget):
        fresh_model.zero_grad(set_to_none=True)
        stats = rl.policy_gradient_backward(
            fresh_model, tokenizer, torch.amp.GradScaler(enabled=False), token_ids,
            completion_mask, old_logprobs, seq_starts, advantages, blocked, 1.0, 0.01, 1.0, budget,
        )
        flat = torch.cat([p.grad.flatten() for p in fresh_model.parameters() if p.grad is not None])
        return flat, stats

    whole, whole_stats = grads(None)
    # Small enough to force several chunks out of 12 rows of up to ~19 tokens
    chunked, chunked_stats = grads(60)
    assert whole.norm() > 0

    assert torch.nn.functional.cosine_similarity(whole, chunked, dim=0) > 0.999
    assert chunked.norm().item() == pytest.approx(whole.norm().item(), rel=0.02)
    assert chunked_stats[0] == pytest.approx(whole_stats[0], rel=0.02, abs=1e-4)
    assert chunked_stats[3] == whole_stats[3]
    assert chunked_stats[2] / chunked_stats[3] < 1e-3


def test_pooled_minibatches_give_the_gradient_of_training_on_each(fresh_model, tokenizer):
    """A step's rollouts are rescored together, whatever minibatches generated them.

    The two minibatches decode to different widths, so pooling has to pad one.
    """
    from src import generation

    torch.manual_seed(0)
    minibatches = []
    for prompts, max_new_tokens in ((["hello", "foo bar baz"], 4), (["the quick brown fox jumps over"], 12)):
        prompts = [p for p in prompts for _ in range(4)]
        token_ids, completion_mask, old_logprobs, seq_starts, _ = generation.generate(
            fresh_model, tokenizer, prompts, DEV, max_new_tokens=max_new_tokens, temperature=1.0
        )
        advantages = torch.randn(len(prompts), device=DEV)
        minibatches.append((token_ids, completion_mask, old_logprobs, seq_starts, advantages))
    assert minibatches[0][0].shape[1] != minibatches[1][0].shape[1]
    blocked = generation.blocked_token_ids(tokenizer, torch.device(DEV))
    fresh_model.train()

    def grads(groups):
        fresh_model.zero_grad(set_to_none=True)
        stats = [
            rl.policy_gradient_backward(
                fresh_model, tokenizer, torch.amp.GradScaler(enabled=False), *group, blocked, 1.0, 0.01, 1.0,
            )
            for group in groups
        ]
        flat = torch.cat([p.grad.flatten() for p in fresh_model.parameters() if p.grad is not None])
        return flat, [sum(column) for column in zip(*stats)]

    separate, separate_stats = grads(minibatches)
    pooled, pooled_stats = grads([rl.pool_rollouts(minibatches, tokenizer.pad_token_id)])
    assert separate.norm() > 0

    assert torch.nn.functional.cosine_similarity(separate, pooled, dim=0) > 0.999
    assert pooled.norm().item() == pytest.approx(separate.norm().item(), rel=0.02)
    assert pooled_stats[0] == pytest.approx(separate_stats[0], rel=0.02, abs=1e-4)
    assert pooled_stats[3] == separate_stats[3]
    assert pooled_stats[2] / pooled_stats[3] < 1e-3


def test_a_step_that_is_not_whole_minibatches_is_refused(fresh_model, tokenizer, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="multiple of generation_prompts_size"):
        _run(
            fresh_model, tokenizer, tmp_path, monkeypatch, _space_led, num_steps=1,
            generation_prompts_size=3,
        )
