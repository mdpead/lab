import argparse
import json
import random
from typing import Any

# Pass-rate gap between two evaluate.py runs, with a 95% bootstrap interval over functions.
# Prompts sharing a function succeed or fail together, so resample functions, not prompts.


def load(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def by_function(result: dict[str, Any]) -> list[list[int]]:
    groups: dict[float, list[int]] = {}
    for i, source in enumerate(result["sources"]):
        groups.setdefault(source, []).append(i)
    return list(groups.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", help="evaluate.py result, e.g. models/base_lr3/eval/sft_10000.json")
    parser.add_argument("b", help="the result to compare it against")
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()

    a, b = load(args.a), load(args.b)
    if a["sources"] != b["sources"]:
        raise ValueError("The two results were scored on different prompts.")

    groups = by_function(a)
    diffs = [
        (sum(a["passed"][i] for i in g) - sum(b["passed"][i] for i in g), len(g)) for g in groups
    ]

    rng = random.Random(0)
    gaps = []
    for _ in range(args.samples):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        gaps.append(sum(d for d, _ in sample) / sum(n for _, n in sample))
    gaps.sort()

    gap = a["pass_rate"] - b["pass_rate"]
    low, high = gaps[int(0.025 * len(gaps))], gaps[int(0.975 * len(gaps))]
    print(
        f"{a['config']} {a['stage']}: {a['pass_rate']:.1%}  {b['config']} {b['stage']}: "
        f"{b['pass_rate']:.1%}  gap {gap:+.1%} (95% CI {low:+.1%} to {high:+.1%}), "
        f"{a['num_prompts']} prompts over {len(groups)} functions"
    )


if __name__ == "__main__":
    main()
