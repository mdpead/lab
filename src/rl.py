import torch
import torch.nn.functional as F
from torch import amp
import logging
import time

from src import train, generation, synth
from src.dataloader import resume_at_minibatch

# Which checker scores a rollout, keyed by the row's task (mirrors data._TASK_FORMATTERS).
_TASK_CHECKERS = {
    "output": synth.check_output_answer,
    "input": synth.check_input_answer,
    "search": synth.check_input_answer,
}

# Tasks trained on a graded reward. Validation still reports the checker's verdict.
_TASK_REWARDS = {
    "search": synth.reward_search_answer,
}


# Extra per-rollout figures a task reports at validation, averaged over greedy rollouts.
_TASK_STATS = {
    "search": synth.search_stats,
}


def score(rows, completions):
    """Reward each rollout 1.0 if its checker accepts the answer, else 0.0."""
    return [float(_TASK_CHECKERS[row["task"]](row, c)) for row, c in zip(rows, completions)]


def reward(rows, completions):
    """The reward each rollout is trained on: its task's graded reward, else its score."""
    return [
        float(_TASK_REWARDS.get(row["task"], _TASK_CHECKERS[row["task"]])(row, c))
        for row, c in zip(rows, completions)
    ]


def validation_step(
    model,
    dataloader,
    tokenizer,
    device,
    step_no,
    max_new_tokens,
    max_batches,
    samples_per_prompt,
    temperature,
    seed,
):
    """Score the test prompts two ways: sampled reward and greedy pass@1.

    `sampled_reward` is what the policy gradient raises directly, so it moves first.
    `pass_rate` is the number that matters in the end, and lags it.

    Sampling runs under a fixed seed, restored afterwards, so validations are comparable
    and training's own draws are left alone.
    """
    sampled, passed, finished_count, total = 0.0, 0.0, 0, 0
    stats = {}
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            # Sampled: each prompt repeated, as training builds its groups
            rows = [row for row in batch for _ in range(samples_per_prompt)]
            token_ids, _, _, _, _ = generation.generate(
                model,
                tokenizer,
                [row["prompt"] for row in rows],
                device,
                max_new_tokens,
                temperature,
            )
            sampled += sum(score(rows, generation.decode_rollouts(tokenizer, token_ids)))

            # Greedy
            texts, finished = generation.generate_texts(
                model, tokenizer, [row["prompt"] for row in batch], device, max_new_tokens
            )
            passed += sum(score(batch, texts))
            for row, text in zip(batch, texts):
                if row["task"] in _TASK_STATS:
                    for key, value in _TASK_STATS[row["task"]](row, text).items():
                        stats.setdefault(key, []).append(value)
            finished_count += sum(finished)
            total += len(batch)

    return {
        "type": "validation",
        "step_no": step_no,
        "sampled_reward": sampled / max(total * samples_per_prompt, 1),
        "pass_rate": passed / max(total, 1),
        "finished_rate": finished_count / max(total, 1),
        "num_prompts": total,
        **{key: sum(values) / len(values) for key, values in stats.items()},
    }


def token_budget_chunks(starts, ends, budget):
    """Split rows into chunks whose padded size, rows x width, fits `budget` tokens.

    `starts` and `ends` are each row's first real column and one past its last. Rollout
    lengths are only known after generation, so this cannot live in the sampler.

    Rows are taken in order of `ends`, so short rollouts share a chunk. A row wider than
    the budget still gets its own chunk: dropping it would bias the update towards short
    completions. `budget` None means one chunk.
    """
    order = sorted(range(len(ends)), key=ends.__getitem__)
    if budget is None:
        return [order]

    chunks, current = [], []
    for row in order:
        candidate = current + [row]
        width = max(ends[i] for i in candidate) - min(starts[i] for i in candidate)
        if current and len(candidate) * width > budget:
            chunks.append(current)
            current = [row]
        else:
            current = candidate
    chunks.append(current)
    return chunks


def pool_rollouts(minibatches, pad_token_id):
    """Join the rollouts of several generation minibatches into one set of tensors.

    Each minibatch is (token_ids, completion_mask, old_logprobs, seq_starts, advantages).
    Narrower ones are right-padded to the widest, the side generation already pads on.
    """
    width = max(token_ids.shape[1] for token_ids, *_ in minibatches)
    fills = (pad_token_id, False, 0.0)
    padded = [
        [F.pad(t, (0, width - t.shape[1]), value=fill) for t, fill in zip(minibatch, fills)]
        for minibatch in minibatches
    ]
    return (
        *(torch.cat(column) for column in zip(*padded)),
        torch.cat([minibatch[3] for minibatch in minibatches]),
        torch.cat([minibatch[4] for minibatch in minibatches]),
    )


def policy_gradient_backward(
    model,
    tokenizer,
    scaler,
    token_ids,
    completion_mask,
    old_logprobs,
    seq_starts,
    advantages,
    blocked,
    temperature,
    router_aux_loss_coef,
    loss_scale,
    minibatch_token_size=None,
):
    """Rescore rollouts with gradients on and backpropagate the policy-gradient loss.

    Per rollout the loss is -(advantage x mean logprob of its own tokens), plus the router's
    load-balancing term. The sum over rollouts, times `loss_scale`, is backpropagated in
    chunks that fit `minibatch_token_size`, which changes peak memory and not the gradient.

    Returns (loss_pg_sum, loss_aux_sum, logprob_gap_sum, completion_tokens) as floats.
    """
    device = token_ids.device
    length = token_ids.shape[1]

    # One past each row's last real token. Right padding only follows a finished row, and
    # sampling cannot emit <pad>, so the trailing pads are exactly the filler.
    is_pad = token_ids == tokenizer.pad_token_id
    ends = (length - is_pad.flip(1).int().cumprod(1).sum(1)).tolist()
    starts = seq_starts.tolist()

    loss_pg_total, loss_aux_total, gap_total, tokens_total = 0.0, 0.0, 0.0, 0
    for chunk in token_budget_chunks(starts, ends, minibatch_token_size):
        left = min(starts[i] for i in chunk)
        right = max(ends[i] for i in chunk)
        rows = torch.tensor(chunk, device=device)

        ids = token_ids[rows, left:right]
        chosen = completion_mask[rows, left:right]
        chunk_starts = seq_starts[rows] - left

        # seq_starts and padding_mask reproduce the context generation saw. output_mask
        # projects only the positions whose next token the model chose.
        predicts_completion = F.pad(chosen[:, 1:], (0, 1), value=False)
        with amp.autocast(device_type=device.type):
            logits, loss_aux = model(
                ids,
                seq_starts=chunk_starts,
                padding_mask=ids != tokenizer.pad_token_id,
                output_mask=predicts_completion,
            )  # (completion tokens, vocab), row-major

        # fp32, tempered and with the blocked ids removed, exactly as generate sampled:
        # the gradient needs the logprobs of the distribution the rollouts were drawn from.
        step_logits = logits.float() / temperature
        step_logits[:, blocked] = -float("inf")
        # Boolean indexing is row-major on both sides, so these line up with the logits
        targets = ids[:, 1:][chosen[:, 1:]]
        new_logprobs = -F.cross_entropy(step_logits, targets, reduction="none")  # (tokens,)

        # Mean over each rollout's own tokens, so a long completion does not outweigh a
        # short one. Summed over rollouts, not averaged: the caller scales.
        row_of_token = predicts_completion.nonzero()[:, 0]
        tokens_per_row = predicts_completion.sum(-1)
        per_seq = torch.zeros(len(chunk), device=device).index_add_(0, row_of_token, new_logprobs)
        per_seq = per_seq / tokens_per_row.clamp(min=1)
        loss_pg = -(per_seq * advantages[rows]).sum()
        # The aux term is a per-pass mean, so it is weighted by the rows that pass held
        loss = loss_pg + loss_aux * len(chunk) * router_aux_loss_coef

        scaler.scale(loss * loss_scale).backward()

        with torch.no_grad():
            old = old_logprobs[rows, left:right][:, 1:][chosen[:, 1:]]
            gap_total += (new_logprobs - old).abs().sum().item()
        loss_pg_total += loss_pg.item()
        loss_aux_total += loss_aux.item() * len(chunk)
        tokens_total += int(tokens_per_row.sum().item())

    return loss_pg_total, loss_aux_total, gap_total, tokens_total


def train_loop(stage, model, dataloaders, tokenizer, run, config):

    train_config = config["train"][stage]
    device = torch.device(train_config["device"])
    # A step is defined in prompts; generation_prompts_size only batches their generation
    if train_config["effective_prompts_size"] % train_config["generation_prompts_size"] != 0:
        raise ValueError(
            f"effective_prompts_size ({train_config['effective_prompts_size']}) must be a "
            f"multiple of generation_prompts_size ({train_config['generation_prompts_size']})"
        )
    minibatches_per_step = (
        train_config["effective_prompts_size"] // train_config["generation_prompts_size"]
    )
    num_steps = train_config["num_steps"]
    checkpoint_steps = train_config["checkpoint_steps"]
    keep_checkpoints = train_config.get("keep_checkpoints")
    validation_steps = train_config["validation_steps"]
    validation_batches = train_config["validation_batches"]
    cache_clear_steps = train_config.get("cache_clear_steps")
    router_aux_loss_coef = train_config["router_aux_loss_coef"]
    completions_per_prompt = train_config["completions_per_prompt"]
    max_new_tokens = train_config["max_new_tokens"]
    temperature = train_config["temperature"]
    minibatch_token_size = train_config.get("minibatch_token_size")
    advantage_sd_floor = train_config.get("advantage_sd_floor", 1e-8)

    optimiser = run["optimiser"]
    lr_scheduler = run["lr_scheduler"]
    scaler = run["scaler"]
    results = run["results"]
    step_no = run["step_no"]
    run_path = run["run_path"]

    blocked = generation.blocked_token_ids(tokenizer, device)

    def validate():
        result = validation_step(
            model,
            dataloaders["test"],
            tokenizer,
            device,
            step_no,
            max_new_tokens,
            # In validation batches, which have their own size
            validation_batches,
            train_config.get("validation_samples", completions_per_prompt),
            temperature,
            config["seed"],
        )
        logging.info(result)
        results.append(result)
        model.train()

    # Record the starting point on a fresh run, for later validations to compare against
    if step_no == 0 and not any(r["type"] == "validation" for r in results):
        validate()

    start_time = time.time()
    total_loss_pg = 0.0
    total_loss_aux = 0.0
    total_reward = 0.0
    total_solved = 0.0
    total_finished = 0.0
    total_mixed_groups = 0.0
    total_logprob_gap = 0.0
    total_completion_tokens = 0
    total_rescored_tokens = 0
    trained_groups = 0
    # Rollouts from mixed groups, held until the whole step is generated
    pending = []
    resume_at_minibatch(dataloaders["train"], step_no * minibatches_per_step)
    optimiser.zero_grad(set_to_none=True)

    model.train()
    for minibatch_idx, batch in enumerate(dataloaders["train"]):

        # One row per rollout, each prompt's group contiguous for the reshape below
        rows = [row for row in batch for _ in range(completions_per_prompt)]
        prompts = [row["prompt"] for row in rows]

        # Sample the rollouts, under no_grad
        token_ids, completion_mask, old_logprobs, seq_starts, finished = generation.generate(
            model, tokenizer, prompts, device, max_new_tokens, temperature
        )

        # Scoring executes the record that produced the prompt, so the whole row is passed
        completions = generation.decode_rollouts(tokenizer, token_ids)
        rewards = torch.tensor(reward(rows, completions), dtype=torch.float32, device=device)
        rewards = rewards.reshape(len(batch), completions_per_prompt)
        solved = sum(score(rows, completions)) / len(rows)

        # Group-relative advantage. The floor stops a group whose graded rewards barely
        # differ from being normalised up to a full-size advantage.
        mean = rewards.mean(dim=-1, keepdim=True)
        sd = rewards.std(dim=-1, keepdim=True)
        advantages = ((rewards - mean) / sd.clamp(min=advantage_sd_floor)).reshape(-1)

        # A uniform group has zero advantage and so zero gradient: drop it rather than
        # rescore it, and keep it out of the average.
        mixed = sd.squeeze(-1) > 0  # (prompts,)
        keep = mixed.repeat_interleave(completions_per_prompt).nonzero().squeeze(-1)

        # Held for the step's one rescore
        if len(keep) > 0:
            pending.append(
                (
                    token_ids[keep],
                    completion_mask[keep],
                    old_logprobs[keep],
                    seq_starts[keep],
                    advantages[keep],
                )
            )
            trained_groups += int(mixed.sum().item())

        # Over every group, dropped or not
        total_reward += rewards.mean().item() / minibatches_per_step
        total_solved += solved / minibatches_per_step
        total_finished += finished.float().mean().item() / minibatches_per_step
        total_mixed_groups += mixed.float().mean().item() / minibatches_per_step
        total_completion_tokens += int(completion_mask.sum().item())

        # Keep generating until the step has all its prompts
        if (minibatch_idx + 1) % minibatches_per_step != 0:
            continue

        # Rescore and backpropagate the whole step's rollouts, in chunks that fit the token
        # budget, averaged over the rollouts trained on. A step with no mixed group leaves
        # the weights alone but still counts, so step_no tracks the sampler's position.
        trained_rollouts = trained_groups * completions_per_prompt
        if trained_rollouts > 0:
            total_loss_pg, total_loss_aux, total_logprob_gap, total_rescored_tokens = (
                policy_gradient_backward(
                    model,
                    tokenizer,
                    scaler,
                    *pool_rollouts(pending, tokenizer.pad_token_id),
                    blocked,
                    temperature,
                    router_aux_loss_coef,
                    1 / trained_rollouts,
                    minibatch_token_size,
                )
            )
            scaler.unscale_(optimiser)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            scaler.step(optimiser)
            scaler.update()
        else:
            grad_norm = 0.0
        lr_scheduler.step()
        optimiser.zero_grad(set_to_none=True)

        step_no += 1

        elapsed_time = time.time() - start_time
        result = {}
        result["type"] = "train"
        result["step_no"] = step_no
        result["num_completion_tokens"] = total_completion_tokens
        result["tokens_per_sec"] = total_completion_tokens / elapsed_time
        result["learning_rate"] = lr_scheduler.get_last_lr()[0]
        # Losses and the gap are over the rollouts trained on; the rest over all of them
        result["loss_pg"] = total_loss_pg / max(trained_rollouts, 1)
        result["loss_aux"] = total_loss_aux / max(trained_rollouts, 1)
        result["reward"] = total_reward
        result["solved_rate"] = total_solved
        result["finished_rate"] = total_finished
        result["mixed_group_rate"] = total_mixed_groups
        result["trained_groups"] = trained_groups
        # Rescore against the sampling pass on the same tokens: should sit near zero
        result["logprob_gap"] = total_logprob_gap / max(total_rescored_tokens, 1)
        result["grad_norm"] = grad_norm
        results.append(result)
        logging.info(result)

        # Validation step
        if step_no % validation_steps == 0:
            validate()

        # Checkpointing
        if step_no % checkpoint_steps == 0:
            train.save_checkpoint(
                model, optimiser, lr_scheduler, scaler, run_path, step_no, results, keep_checkpoints
            )

        # Delete tensors to free up memory
        del token_ids, completion_mask, old_logprobs
        if cache_clear_steps and step_no % cache_clear_steps == 0:
            torch.cuda.empty_cache()

        # Reset counters
        start_time = time.time()
        total_loss_pg = 0.0
        total_loss_aux = 0.0
        total_reward = 0.0
        total_solved = 0.0
        total_finished = 0.0
        total_mixed_groups = 0.0
        total_logprob_gap = 0.0
        total_completion_tokens = 0
        total_rescored_tokens = 0
        trained_groups = 0
        pending = []

        # Stop after num_steps
        if step_no >= num_steps:
            break

    # A final save when num_steps is not a multiple of checkpoint_steps
    if step_no % checkpoint_steps != 0:
        train.save_checkpoint(
            model, optimiser, lr_scheduler, scaler, run_path, step_no, results, keep_checkpoints
        )

    return None


def train_rl(stage, model, dataloaders, tokenizer, config, init_from=None):
    """Run the RL stage: train.train's bookkeeping, this module's loop."""
    return train.train(stage, model, dataloaders, tokenizer, config, init_from, loop=train_loop)
