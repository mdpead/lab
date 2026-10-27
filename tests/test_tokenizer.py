"""Saving and loading tokenizers, and clearing a stale one when its config changes."""

import os

import pytest

from mlcore import runs
from mlcore import tokenizer as tokenizer_utils


def _run_config(tmp_path, vocab_size=100):
    return {
        "name": "t",
        "seed": 0,
        "locations": {"models_dir": str(tmp_path)},
        "tokenizer": {"vocab_size": vocab_size},
    }


def test_loading_a_tokenizer_that_was_never_trained_says_how_to_train_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts.tokenizer"):
        tokenizer_utils.load_tokenizer(str(tmp_path))


def test_a_changed_tokenizer_config_clears_the_saved_tokenizer(tmp_path):
    """Stages only load the tokenizer, so a stale one must not survive a config change."""
    run_path = runs.init_run_dir(_run_config(tmp_path))
    os.makedirs(f"{run_path}/tokenizer")

    runs.init_run_dir(_run_config(tmp_path))
    assert os.path.isdir(f"{run_path}/tokenizer")

    runs.init_run_dir(_run_config(tmp_path, vocab_size=200))
    assert not os.path.isdir(f"{run_path}/tokenizer")
