import warnings

import torch
import torch.nn.functional as F


def make_kv_cache(model, batches, seq_length, device):

    d_h = model.d_h
    num_layers = model.num_layers
    num_kv_heads = model.num_kv_heads
    dtype = next(model.parameters()).dtype

    kv_cache = torch.zeros(
        (num_layers, 2, batches, num_kv_heads, seq_length, d_h), device=device, dtype=dtype
    )

    return kv_cache


def blocked_token_ids(tokenizer, device):
    """Ids sampling never draws: <bos>, <pad> and <unk>. <eos> is allowed, so the model can stop.

    Shared with the policy gradient, which must rescore under the same distribution.
    """
    blocked = [tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id]
    return torch.tensor([i for i in blocked if i is not None], device=device)


def generate(model, tokenizer, input_texts, device, max_new_tokens=None, temperature=0):
    """Continue each prompt, returning (token_ids, completion_mask, logprobs, seq_starts,
    finished).

    - `token_ids` (batch, length): <bos>, prompt and continuation, left-padded so prompts
      end on the same column and right-padded after a row stops.
    - `completion_mask`: True where the model chose the token, its <eos> included.
    - `logprobs`: log-probability of each drawn token under the sampling distribution.
      None under a greedy decode, so training on it fails rather than computing nonsense.
    - `seq_starts`: each row's first real column, needed to feed the ids back to the model.
    - `finished`: True when a row stopped on <eos> rather than running out of budget.

    `temperature` 0 means greedy, which is reproducible and independent of batching.
    """
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")

    max_length = model.max_length

    # Encode without special tokens so we control them: <bos> leads, no trailing <eos>
    prompt_ids = tokenizer(input_texts, add_special_tokens=False)["input_ids"]
    prompt_ids = [[tokenizer.bos_token_id] + ids for ids in prompt_ids]

    # A prompt over the context cannot be generated from, so refuse rather than truncate
    over = [len(ids) for ids in prompt_ids if len(ids) > max_length]
    if over:
        raise ValueError(
            f"{len(over)} prompt(s) exceed max_length={max_length} (longest is {max(over)})"
        )
    encoded = tokenizer.pad({"input_ids": prompt_ids}, return_tensors="pt", padding_side="left")

    input_ids = encoded["input_ids"].to(device)
    batch_size, input_length = input_ids.shape

    # Whatever the prompt leaves is the generation budget, optionally capped further
    room = max_length - input_length
    new_tokens = room if max_new_tokens is None else min(max_new_tokens, room)
    if new_tokens < 1:
        # The prompt fills the context: hand back the prompts unchanged
        warnings.warn(f"prompt fills {input_length} of {max_length} tokens, generating nothing")
    elif max_new_tokens is not None and new_tokens < max_new_tokens:
        warnings.warn(
            f"max_new_tokens reduced from {max_new_tokens} to {new_tokens}; "
            f"the prompt uses {input_length} of {max_length} tokens"
        )

    # Left padding gives the batch one shared frontier
    seq_starts = (input_length - encoded["attention_mask"].sum(dim=-1)).to(
        device=device, dtype=torch.int32
    )  # (batch,) first real column
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Preallocate the whole buffer, then fill forward from the prompt
    token_ids = F.pad(input_ids, (0, new_tokens), value=tokenizer.pad_token_id)
    completion_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    logprobs = torch.zeros_like(token_ids, dtype=torch.float32) if temperature > 0 else None

    blocked = blocked_token_ids(tokenizer, device)

    kv_cache = make_kv_cache(model, batch_size, input_length + new_tokens, device)

    num_cached = 0
    cur_length = input_length
    training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(new_tokens):

            kv_cache_step = kv_cache[..., 0:num_cached, :]
            token_ids_step = token_ids[:, num_cached:cur_length]

            logits, _, new_kv = model(token_ids_step, seq_starts, kv_cache_step)
            kv_cache[..., num_cached:cur_length, :] = new_kv

            # Choose the next token
            if temperature > 0:
                # fp32: bf16 collapses the tail of the vocabulary and skews the draw
                step_logits = logits[:, -1, :].float()
                step_logits[:, blocked] = -float("inf")
                step_logprobs = F.log_softmax(step_logits / temperature, dim=-1)
                pred_token_ids = torch.multinomial(step_logprobs.exp(), num_samples=1).squeeze(-1)
                # Gathered before the <pad> overwrite below, whose logprob is -inf and
                # would turn a masked sum into nan
                logprobs[:, cur_length] = step_logprobs.gather(
                    1, pred_token_ids[:, None]
                ).squeeze(1)
            else:
                pred_token_ids = logits[:, -1, :].argmax(dim=-1)  # (batch,)
            pred_token_ids = torch.where(finished, tokenizer.pad_token_id, pred_token_ids)

            token_ids[:, cur_length] = pred_token_ids
            # finished is the state before this token, so a row's own <eos> counts as chosen
            completion_mask[:, cur_length] = ~finished
            num_cached = cur_length
            cur_length += 1

            finished |= pred_token_ids == tokenizer.eos_token_id
            if finished.all():
                break

    model.train(training)

    # Trim to what was written: an all-finished batch stops before the buffer is full
    if logprobs is not None:
        logprobs = logprobs[:, :cur_length]
    return (
        token_ids[:, :cur_length],
        completion_mask[:, :cur_length],
        logprobs,
        seq_starts,
        finished,
    )


def decode_rollouts(tokenizer, token_ids):
    """Decode rows of token ids to text, dropping only <bos>, <pad> and <eos>.

    Filtered by hand rather than with skip_special_tokens, which would also strip the
    reasoning markers callers split on. A stray <eos> would break parsing of the answer.
    """
    framing = {tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id}
    return [
        tokenizer.decode([i for i in row if i not in framing], skip_special_tokens=False)
        for row in token_ids.tolist()
    ]


def generate_texts(model, tokenizer, input_texts, device, max_new_tokens=None):
    """Greedy-decode each prompt, returning (texts, finished).

    Each text is the prompt plus its continuation. `finished[i]` is True when row i
    stopped on <eos> rather than running out of budget.
    """
    token_ids, _, _, _, finished = generate(
        model, tokenizer, input_texts, device, max_new_tokens
    )

    texts = decode_rollouts(tokenizer, token_ids)

    return texts, finished.tolist()
