import argparse
import re
import sys
import unicodedata
from collections import defaultdict

from datasets import load_dataset

sys.path.insert(0, ".")

from mlcore import runs

from src import datasets as ds_module
from src.config import Config
from src.train import STAGE

N = 8  # n-gram length for partial-overlap detection
COVERAGE_THRESHOLD = 0.5

parser = argparse.ArgumentParser(
    description="Check FLORES+ devtest for overlap with the training split"
)
parser.add_argument("--config", required=True, help="Config name (e.g. base)")
args = parser.parse_args()


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def ngrams(tokens: list[str]) -> set[str]:
    return {" ".join(tokens[i : i + N]) for i in range(len(tokens) - N + 1)}


config = runs.load_config_file(f"configs/{args.config}.yaml", Config)

train = ds_module.get_dataset(STAGE, config)["train"]
print(f"Training split: {len(train)} pairs")

flores = load_dataset("openlanguagedata/flores_plus")["devtest"]
flores_text = {
    "en": flores.filter(lambda x: x["iso_639_3"] == "eng")["text"],
    "cy": flores.filter(lambda x: x["iso_639_3"] == "cym")["text"],
}
print(f"FLORES+ devtest: {len(flores_text['en'])} sentences")

for side, col in (("en", "text_en"), ("cy", "text_cy")):
    flores_raw = flores_text[side]
    flores_norm = [norm(s) for s in flores_raw]
    train_texts = [t for t in train[col] if t]

    # Exact and normalised matches
    raw_set = {t.strip() for t in train_texts}
    norm_set = {norm(t) for t in train_texts}
    exact = sum(1 for s in flores_raw if s.strip() in raw_set)
    normed = sum(1 for s in flores_norm if s in norm_set)

    # Partial overlap: share of each FLORES+ sentence's n-grams found anywhere in the training text
    gram_owner: defaultdict[str, set[int]] = defaultdict(set)
    flores_grams: list[set[str]] = []
    for i, s in enumerate(flores_norm):
        g = ngrams(s.split())
        flores_grams.append(g)
        for x in g:
            gram_owner[x].add(i)
    hits: defaultdict[int, set[str]] = defaultdict(set)
    for t in train_texts:
        for x in ngrams(norm(t).split()):
            for i in gram_owner.get(x, ()):
                hits[i].add(x)
    coverage = {i: len(hits[i]) / len(g) for i, g in enumerate(flores_grams) if g}
    high = sorted(
        (i for i, c in coverage.items() if c >= COVERAGE_THRESHOLD), key=lambda i: -coverage[i]
    )
    any_shared = sum(1 for c in coverage.values() if c > 0)

    print(f"\n=== {side.upper()} ===")
    print(f"Exact matches:                {exact}")
    print(f"Normalised matches:           {normed}")
    print(f">={COVERAGE_THRESHOLD:.0%} {N}-gram overlap:         {len(high)}")
    print(f"Sharing any {N}-gram:           {any_shared}")
    for i in high[:15]:
        print(f"  [{i}] {coverage[i]:.0%}  {flores_raw[i][:150]}")
