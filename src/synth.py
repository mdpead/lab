import ast
import random
import re
import sys

# Values stay small so the task is hard because of control flow, not arithmetic. Programs
# that leave the range are rejected after execution rather than clamped.
VALUE_MIN, VALUE_MAX = -500, 500
MAX_RESULT_CHARS = 80

# Named per nesting depth so an inner loop never shadows its parent
LOOP_VARS = ["i", "j", "k"]
INT_VARS = ["a", "b", "c", "d"]
LIST_VARS = ["xs", "ys"]

# max_executed caps executed lines, bounding trace length. Retune against the tokenizer:
# the budget that matters is trace tokens, and RL needs headroom above it.
TIERS = {
    # max_depth 0 is straight-line code, which a freshly pretrained model can do
    "easy": dict(n_stmts=(3, 5), max_depth=0, n_ints=2, n_lists=0, loop_range=(2, 3), max_executed=25),
    "medium": dict(n_stmts=(4, 7), max_depth=1, n_ints=2, n_lists=1, loop_range=(2, 4), max_executed=45),
    "hard": dict(n_stmts=(6, 10), max_depth=2, n_ints=3, n_lists=2, loop_range=(2, 5), max_executed=70),
}


class _TooLong(Exception):
    """Raised from inside the tracer to abandon an over-long run mid-execution."""


def _gen_expr(rng, scope, depth=0):
    # depth is expression nesting, not statement nesting
    choices = ["int_var", "literal"]
    if scope["lists"]:
        choices += ["index", "len"]
    if depth < 2:
        choices += ["binop", "binop"]

    kind = rng.choice(choices)
    if kind == "int_var":
        return rng.choice(scope["ints"])
    if kind == "literal":
        return str(rng.randint(-5, 9))
    if kind == "len":
        return f"len({rng.choice(scope['lists'])})"
    if kind == "index":
        # Lists are only appended to, so they are non-empty and the modulo stays in bounds
        name = rng.choice(scope["lists"])
        return f"{name}[{rng.choice(scope['ints'])} % len({name})]"
    left = _gen_expr(rng, scope, depth + 1)
    right = _gen_expr(rng, scope, depth + 1)
    return f"({left} {rng.choice(['+', '-', '*'])} {right})"


def _gen_cond(rng, scope):
    op = rng.choice(["<", ">", "<=", ">=", "==", "!="])
    return f"{_gen_expr(rng, scope)} {op} {rng.randint(-5, 15)}"


def _gen_stmt(rng, scope, depth, cfg, indent):
    pad = "    " * indent
    kinds = ["assign", "assign", "augmented"]
    if scope["lists"]:
        kinds.append("append")
    if depth < cfg["max_depth"]:
        kinds += ["if", "for"]

    kind = rng.choice(kinds)
    if kind == "assign":
        return [f"{pad}{rng.choice(scope['ints'])} = {_gen_expr(rng, scope)}"]
    if kind == "augmented":
        op = rng.choice(["+=", "-=", "*="])
        return [f"{pad}{rng.choice(scope['ints'])} {op} {_gen_expr(rng, scope)}"]
    if kind == "append":
        return [f"{pad}{rng.choice(scope['lists'])}.append({_gen_expr(rng, scope)})"]
    if kind == "if":
        lines = [f"{pad}if {_gen_cond(rng, scope)}:"]
        lines += _gen_block(rng, scope, depth + 1, cfg, indent + 1)
        if rng.random() < 0.4:
            lines.append(f"{pad}else:")
            lines += _gen_block(rng, scope, depth + 1, cfg, indent + 1)
        return lines

    var = LOOP_VARS[depth]
    # range() evaluates its bound once, so a body that appends to the list still terminates
    if scope["lists"] and rng.random() < 0.4:
        bound = f"len({rng.choice(scope['lists'])})"
    else:
        bound = str(rng.randint(*cfg["loop_range"]))
    lines = [f"{pad}for {var} in range({bound}):"]
    inner = dict(scope, ints=scope["ints"] + [var])
    lines += _gen_block(rng, inner, depth + 1, cfg, indent + 1)
    return lines


def _gen_block(rng, scope, depth, cfg, indent):
    n = rng.randint(1, 2) if depth else rng.randint(*cfg["n_stmts"])
    lines = []
    for _ in range(n):
        lines += _gen_stmt(rng, scope, depth, cfg, indent)
    return lines


def _gen_function(rng, cfg):
    """Build a function whose parameters are the task's input slot."""
    ints = INT_VARS[: cfg["n_ints"]]
    lists = LIST_VARS[: cfg["n_lists"]]
    scope = {"ints": ints, "lists": lists}
    params = ints + lists

    lines = [f"def f({', '.join(params)}):"]
    lines += _gen_block(rng, scope, 0, cfg, 1)
    returned = rng.sample(params, k=min(2, len(params)))
    lines.append(f"    return {', '.join(returned)}")
    return "\n".join(lines), [(n, "list" if n in lists else "int") for n in params]


def _sample_args(rng, params):
    args = []
    for _, kind in params:
        if kind == "int":
            args.append(rng.randint(-3, 9))
        else:
            args.append([rng.randint(0, 9) for _ in range(rng.randint(2, 5))])
    return args


def _snapshot(frame):
    # Lists are copied, because append mutates in place
    return {
        k: (list(v) if isinstance(v, list) else v)
        for k, v in frame.f_locals.items()
        if not k.startswith("__")
    }


def _run_traced(source, code_obj, func, args, max_executed):
    """Call func(*args) under a tracer, returning (result, trace_steps, n_executed).

    Returns None if the call raises or any value escapes the allowed range.
    """
    src_lines = source.split("\n")
    steps = []
    prev = {}
    overflowed = []

    def tracer(frame, event, arg):
        if frame.f_code is not code_obj:
            return None
        if event == "call":
            # Seed from the parameters so the trace does not restate the arguments
            prev.update(_snapshot(frame))
            return tracer
        if event == "line":
            # A line event fires before the line runs, so this state is the previous line's
            current = _snapshot(frame)
            changed = {k: v for k, v in current.items() if prev.get(k, object()) != v}
            for value in changed.values():
                flat = value if isinstance(value, list) else [value]
                if any(isinstance(v, int) and not VALUE_MIN <= v <= VALUE_MAX for v in flat):
                    overflowed.append(True)
            if changed and steps:
                shown = ", ".join(f"{k} = {v!r}" for k, v in sorted(changed.items()))
                steps[-1] = f"{steps[-1]}  ->  {shown}"
            prev.clear()
            prev.update(current)
            steps.append(src_lines[frame.f_lineno - 1].strip())
            if len(steps) > max_executed:
                # Abandon an over-long program partway
                raise _TooLong
        return tracer

    sys.settrace(tracer)
    try:
        result = func(*[list(a) if isinstance(a, list) else a for a in args])
    except Exception:
        return None
    finally:
        sys.settrace(None)

    if overflowed or len(repr(result)) > MAX_RESULT_CHARS:
        return None
    return result, steps, len(steps)


# The longest run any tier allows, for tracing outside generate() where there is no tier
_MAX_EXECUTED = max(cfg["max_executed"] for cfg in TIERS.values())


def _trace_call(source, args):
    """Trace f(*args) for a function given as source. _run_traced's result, or None."""
    namespace = {}
    try:
        exec(compile(source, "<synth>", "exec"), namespace)
        func = namespace["f"]
        return _run_traced(source, func.__code__, func, args, _MAX_EXECUTED)
    except Exception:
        return None


def generate(n, tier="medium", seed=0, inputs_per_fn=4, with_trace=True):
    """Yield n records, sharing each generated function across several sampled inputs."""
    # The distinctness filter below can never pass with fewer than two inputs
    if inputs_per_fn < 2:
        raise ValueError(
            f"inputs_per_fn must be at least 2, got {inputs_per_fn}. One input cannot show "
            "that the result depends on the arguments, which is what the task requires."
        )
    rng = random.Random(seed)
    cfg = TIERS[tier]
    seen = set()
    produced = 0

    while produced < n:
        source, params = _gen_function(rng, cfg)
        if source in seen:
            continue
        seen.add(source)

        code_obj = compile(source, "<synth>", "exec")
        namespace = {}
        exec(code_obj, namespace)
        func = namespace["f"]
        # The tracer needs the function's own code object, not the module's
        func_code = func.__code__

        batch = []
        for _ in range(inputs_per_fn):
            args = _sample_args(rng, params)
            outcome = _run_traced(source, func_code, func, args, cfg["max_executed"])
            if outcome is None:
                continue
            result, steps, n_executed = outcome
            batch.append(
                {
                    "source": source,
                    "args": args,
                    "result": result,
                    "tier": tier,
                    # The difficulty signal; varies per input when a loop bound is a list
                    "n_executed": n_executed,
                    **({"trace": "\n".join(steps)} if with_trace else {}),
                }
            )

        # Require that the input actually moves the result: otherwise output prediction
        # is memorisable and input prediction is satisfied by anything.
        if len({repr(r["result"]) for r in batch}) < 2:
            continue

        for record in batch:
            if produced >= n:
                break
            yield record
            produced += 1


def _call_repr(args):
    return f"f({', '.join(repr(a) for a in args)})"


# Single tokens in the trained tokenizer, which cannot occur in generated source
THINK_OPEN, THINK_CLOSE = "<|think|>", "<|/think|>"
ANSWER_OPEN, ANSWER_CLOSE = "<|answer|>", "<|/answer|>"


def _wrap_trace(record, include_trace):
    return f"{THINK_OPEN}{record['trace']}{THINK_CLOSE}" if include_trace else ""


def format_output_task(record, include_trace):
    """Given the function and its arguments, predict the return value."""
    prompt = f"{record['source']}\n\n# What does {_call_repr(record['args'])} return?\n"
    body = _wrap_trace(record, include_trace)
    return prompt, f"{body}{ANSWER_OPEN}{record['result']!r}{ANSWER_CLOSE}"


def format_input_task(record, include_trace):
    """Given the function and a target return value, find arguments that produce it."""
    prompt = f"{record['source']}\n\n# What arguments make f return {record['result']!r}?\n"
    body = _wrap_trace(record, include_trace)
    answer = ", ".join(repr(a) for a in record["args"])
    return prompt, f"{body}{ANSWER_OPEN}{answer}{ANSWER_CLOSE}"


# Opens each attempt in the search task's thinking. No generated statement starts with it.
TRY_PREFIX = "try "
# Most missed guesses shown before the one that works. The count varies per example so
# that stopping follows from reaching the target, not from how many came before.
MAX_MISSES = 2


def _args_repr(args):
    return ", ".join(repr(a) for a in args)


def _format_attempt(args, steps, result):
    """One guess as the search task writes it: the arguments, their trace, what came back."""
    lines = [f"{TRY_PREFIX}{_args_repr(args)}", *steps[:-1], f"{steps[-1]}  ->  {result!r}"]
    return "\n".join(lines)


def _matches(result, target):
    """Fraction of the returned values that equal the target's."""
    if not (isinstance(result, tuple) and isinstance(target, tuple) and len(result) == len(target)):
        return float(result == target)
    return sum(r == t for r, t in zip(result, target)) / max(len(target), 1)


def _perturb(rng, args):
    """Arguments near `args`: at least one changed, each by a small step."""
    out = [list(a) if isinstance(a, list) else a for a in args]
    for i in rng.sample(range(len(out)), k=rng.randint(1, len(out))):
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        if isinstance(out[i], list):
            out[i][rng.randrange(len(out[i]))] += delta
        else:
            out[i] += delta
    return out


def _sample_misses(rng, record, n):
    """Up to n distinct argument sets near the record's own that do not reach its result."""
    misses, seen = [], {repr(record["args"])}
    for _ in range(n * 8):
        if len(misses) == n:
            break
        args = _perturb(rng, record["args"])
        if repr(args) in seen:
            continue
        seen.add(repr(args))
        outcome = _trace_call(record["source"], args)
        if outcome is None or outcome[0] == record["result"]:
            continue
        misses.append((args, outcome[1], outcome[0]))
    # Worst first, so each guess shown is no further from the target than the one before
    return sorted(misses, key=lambda m: _matches(m[2], record["result"]))


def format_search_task(record, include_trace):
    """Input prediction as a search: guess arguments, trace them, try again.

    The thinking is a run of attempts, each a guess, its trace and the value returned,
    ending on the first that returns the target. Misses come from a generator seeded by
    the record, so formatting is deterministic. See miss_argument_spans.
    """
    prompt, completion = format_input_task(record, False)
    if not include_trace:
        return prompt, completion

    rng = random.Random(f"{record['source']}|{record['args']!r}")
    attempts = _sample_misses(rng, record, rng.randint(0, MAX_MISSES))
    attempts.append((record["args"], record["trace"].split("\n"), record["result"]))
    body = "\n".join(_format_attempt(*attempt) for attempt in attempts)
    return prompt, f"{THINK_OPEN}{body}{THINK_CLOSE}{completion}"


_TRY_LINE = re.compile(
    rf"(?:(?<=\n)|(?<={re.escape(THINK_OPEN)})){re.escape(TRY_PREFIX)}(.*)"
)


def miss_argument_spans(completion):
    """Character spans of the arguments each missed attempt in a search completion tries.

    SFT must not be scored on these: trained to predict missed arguments, the model learns
    to open with a wrong guess. Every attempt but the last is a miss.
    """
    return [match.span(1) for match in _TRY_LINE.finditer(completion)][:-1]


def extract_answer(completion):
    """Pull the answer text from a rollout, or None if the model never reached one."""
    if ANSWER_OPEN not in completion:
        return None
    answer = completion.split(ANSWER_OPEN)[-1]
    # Up to the closing marker, or the first line break if the rollout was cut off
    return answer.split(ANSWER_CLOSE)[0].strip().split("\n")[0].strip()


def check_output_answer(record, completion):
    answer = extract_answer(completion)
    if answer is None:
        return False
    try:
        return ast.literal_eval(answer) == record["result"]
    except (ValueError, SyntaxError):
        return False


_NO_RESULT = object()


def _answered_args(completion):
    """The arguments a rollout answered with, as a tuple, or None if it gave none that parse."""
    answer = extract_answer(completion)
    if answer is None:
        return None
    try:
        return ast.literal_eval(f"({answer},)")
    except (ValueError, SyntaxError):
        return None


def _answer_result(record, completion):
    """What the record's function returns for the arguments answered, or _NO_RESULT.

    literal_eval never executes, so only the generator's own function is run.
    """
    args = _answered_args(completion)
    if args is None:
        return _NO_RESULT

    namespace = {}
    exec(compile(record["source"], "<synth>", "exec"), namespace)
    try:
        return namespace["f"](*args)
    except Exception:
        return _NO_RESULT


def check_input_answer(record, completion):
    """Accept any arguments reaching the target, not just the ones used to build the task."""
    return _answer_result(record, completion) == record["result"]


def extract_attempts(completion):
    """The attempts in a search rollout's thinking, each as the text the model wrote for it."""
    if THINK_OPEN not in completion:
        return []
    body = completion.split(THINK_OPEN, 1)[1].split(THINK_CLOSE)[0].split(ANSWER_OPEN)[0]
    attempts = []
    for line in body.strip().split("\n"):
        if line.startswith(TRY_PREFIX):
            attempts.append([line])
        elif attempts:
            attempts[-1].append(line)
    return ["\n".join(lines) for lines in attempts]


def _attempt_args(attempt):
    try:
        return ast.literal_eval(f"({attempt.split(chr(10))[0][len(TRY_PREFIX):]},)")
    except (ValueError, SyntaxError):
        return None


def _is_faithful(record, attempt):
    """Whether an attempt is what the function really does with the arguments it tries.

    Compared as text against the formatter's own rendering of the real trace.
    """
    args = _attempt_args(attempt)
    outcome = _trace_call(record["source"], args) if args is not None else None
    if outcome is None:
        return False
    result, steps, _ = outcome
    return attempt.strip() == _format_attempt(args, steps, result)


def search_stats(record, completion):
    """How a search rollout went, beyond whether its answer was right.

    `first_try_solved` is the figure only better first guesses move: more attempts lift
    the pass rate on their own. The rest say whether the search after a miss is any good,
    and are averaged as counts so that their ratios pool over rollouts:

    - `solved_later`: the answer is right after a first attempt that missed. Over
      1 - first_try_solved, it is how often the search rescues a miss.
    - `improving_attempts` over `later_attempts`: how often an attempt after the first
      lands nearer the target than every attempt before it, decided by running it.
    - `improving_if_blind`: what that count would be for guesses in no particular order,
      where the k-th beats the k - 1 before it one time in k. A search that uses its
      misses beats this; ties count as no improvement, so blind guessing falls short.
    """
    attempts = extract_attempts(completion)
    nearness = []
    for attempt in attempts:
        args = _attempt_args(attempt)
        outcome = _trace_call(record["source"], args) if args is not None else None
        nearness.append(_closeness(outcome[0], record["result"]) if outcome is not None else 0.0)

    first_try_solved = bool(nearness) and nearness[0] == 1.0
    improving = sum(nearness[k] > max(nearness[:k]) for k in range(1, len(nearness)))
    return {
        "attempts": len(attempts),
        "first_try_solved": float(first_try_solved),
        "faithful": sum(_is_faithful(record, a) for a in attempts) / max(len(attempts), 1),
        "solved_later": float(not first_try_solved and check_input_answer(record, completion)),
        "later_attempts": max(len(attempts) - 1, 0),
        "improving_attempts": improving,
        "improving_if_blind": sum(1 / k for k in range(2, len(attempts) + 1)),
    }


def _value_closeness(value, target):
    """How near one returned value is to its target, in [0, 1], and 1 only when equal."""
    if isinstance(value, int) and isinstance(target, int):
        gap = abs(value - target)
        # Relative to the target's size, so missing 100 by 5 counts as nearer than 3 by 5
        return 1 - gap / (gap + abs(target) + 1)
    if isinstance(value, list) and isinstance(target, list):
        # Element by element; a missing or surplus element scores nothing
        pairs = [_value_closeness(v, t) for v, t in zip(value, target)]
        return sum(pairs) / max(len(value), len(target), 1)
    return 0.0


def _closeness(result, target):
    """How near a result is to the target, averaged over the values returned.

    Graded where _matches is all-or-nothing per value, so a guess nearer the target
    scores higher before any value is exactly right.
    """
    if not (isinstance(result, tuple) and isinstance(target, tuple) and len(result) == len(target)):
        return _value_closeness(result, target)
    return sum(_value_closeness(r, t) for r, t in zip(result, target)) / max(len(target), 1)


# Reward terms beside the 1.0 for a correct answer, kept small so they only break ties
REWARD_MATCH = 0.25  # a wrong answer's partial credit, at most
REWARD_FAITHFUL = 0.25  # scaled by the fraction of attempts traced as the function runs
ATTEMPT_COST = 0.02  # per attempt after the first


def _best_closeness(record, completion, attempts):
    """The nearest any guess in a rollout came to the target: its attempts and its answer.

    Decided by running each guess, not by the trace written for it. The best, not the
    last: grading the answer alone made stopping after one guess score better than
    searching on, since a later guess often lands further away, and RL learned to stop.
    Another attempt can only raise this.
    """
    results = [_answer_result(record, completion)]
    for attempt in attempts:
        args = _attempt_args(attempt)
        outcome = _trace_call(record["source"], args) if args is not None else None
        if outcome is not None:
            results.append(outcome[0])
    return max(_closeness(result, record["result"]) for result in results)


def reward_search_answer(record, completion, partial_credit="match"):
    """Graded reward for the search task, all decided by running the function.

    - 1.0 for arguments that reach the target, else REWARD_MATCH times a partial credit:
      "match" is the fraction of returned values the answer gets exactly, "closest" is
      how near the rollout's best guess came (see _best_closeness).
    - REWARD_FAITHFUL for attempts whose trace is the real one. The last attempt counts
      only if the answer repeats its arguments.
    - ATTEMPT_COST off for each attempt after the first.
    """
    if partial_credit not in ("match", "closest"):
        raise ValueError(f"partial_credit must be 'match' or 'closest', got {partial_credit!r}")

    attempts = extract_attempts(completion)
    result = _answer_result(record, completion)
    if result == record["result"]:
        reward = 1.0
    elif partial_credit == "match":
        reward = REWARD_MATCH * _matches(result, record["result"])
    else:
        reward = REWARD_MATCH * _best_closeness(record, completion, attempts)

    if attempts:
        faithful = [_is_faithful(record, attempt) for attempt in attempts]
        if _attempt_args(attempts[-1]) != _answered_args(completion):
            faithful[-1] = False
        reward += REWARD_FAITHFUL * sum(faithful) / len(attempts)
        reward -= ATTEMPT_COST * (len(attempts) - 1)
    return reward
