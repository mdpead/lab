"""The shared tokenizers build what each project's own tokenizer code built before.

The fixture was recorded from that code, on the corpora below, before it was deleted.
"""

import hashlib
import json
from pathlib import Path

import pytest

from lab.tokenizer import train_tokenizer

REFERENCE = json.loads((Path(__file__).parent / "fixtures/tokenizer_reference.json").read_text())

TRANSLATION_PAIRS = [
    ("Hello, how are you?", "Helo, sut wyt ti?"),
    ("Good morning", "Bore da"),
    ("Thank you very much", "Diolch yn fawr iawn"),
    ("The fire is warm", "Mae'r tân yn gynnes"),
    ("Where is the station?", "Ble mae'r orsaf?"),
    ("I live in Cardiff", "Dw i'n byw yng Nghaerdydd"),
    ("Until tomorrow", "Tan yfory"),
    ("The water is cold", "Mae'r dŵr yn oer"),
] * 20

PYTHON_SOURCES = [
    "def add(a, b):\n    return a + b\n",
    "def f(x):\n    y = x * 12345\n    for i in range(10):\n        y += i\n    return y\n",
    "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n",
    "import math\n\nprint(math.sqrt(2) + 3.14159)\n",
    "values = [1, 22, 333, 4444]\nprint(sum(values), max(values))\n",
] * 30

# python_reasoning's markers, in the order its configs list them
MARKERS = ["<|think|>", "<|/think|>", "<|answer|>", "<|/answer|>"]

ROLES = ("bos", "eos", "pad", "unk", "mask")


def _special(tok):
    return {
        role: [getattr(tok, f"{role}_token"), getattr(tok, f"{role}_token_id")] for role in ROLES
    }


def _sha(data):
    return hashlib.sha256(data.encode()).hexdigest()


@pytest.mark.parametrize("digit_split", [False, True])
def test_bpe_matches_python_reasoning_tokenizer(digit_split):
    expected = REFERENCE["bpe"][f"digit_split_{digit_split}"]
    tok = train_tokenizer(
        {
            "type": "bpe",
            "vocab_size": 300,
            "training_size": 120,
            "digit_split": digit_split,
            "special_tokens": MARKERS,
            "reserved_tokens": 16,
        },
        iter(PYTHON_SOURCES),
    )
    assert _sha(tok.backend_tokenizer.to_str()) == expected["sha256"]
    assert len(tok) == expected["vocab_size"]
    assert _special(tok) == expected["special"]
    assert tok("Mae'r tân yn gynnes 12345 def f(x):")["input_ids"] == expected["example_ids"]


@pytest.mark.parametrize("training_size", [None, 100])
def test_wordpiece_matches_en_cy_translation_tokenizer(training_size):
    """Only the structure is compared: WordPiece training breaks ties randomly, so the learned
    vocabulary differs from one process to the next, in the old code as in the new."""
    expected = REFERENCE["wordpiece"][f"training_size_{training_size}"]
    pairs = TRANSLATION_PAIRS
    tok = train_tokenizer(
        {
            "type": "wordpiece",
            "vocab_size": 200,
            "training_size": training_size,
            "strip_accents": True,
        },
        iter([en for en, _ in pairs] + [cy for _, cy in pairs]),
    )
    structure = json.loads(tok.backend_tokenizer.to_str())
    del structure["model"]["vocab"]
    assert _sha(json.dumps(structure, sort_keys=True)) == expected["structure_sha256"]
    assert _special(tok) == expected["special"]
