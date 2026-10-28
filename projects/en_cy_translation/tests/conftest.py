import copy

import pytest
import torch
from mlcore.models import encoder_decoder

from datasets import Dataset, DatasetDict
from src import tokenizer
from src.train import STAGE

PAIRS = [
    ("Hello, how are you?", "Helo, sut wyt ti?"),
    ("Good morning", "Bore da"),
    ("Thank you very much", "Diolch yn fawr iawn"),
    ("I don't know", "Dw i ddim yn gwybod"),
    ("The weather is nice today", "Mae'r tywydd yn braf heddiw"),
    ("Where is the station?", "Ble mae'r orsaf?"),
    ("I live in Cardiff", "Dw i'n byw yng Nghaerdydd"),
    ("See you tomorrow", "Wela i di yfory"),
]

BASE_CONFIG = {
    "name": "pytest",
    "seed": 0,
    "model": {
        "d_model": 32,
        "num_heads": 4,
        "max_length": 32,
        "d_ff": 64,
        "num_enc_layers": 2,
        "num_dec_layers": 2,
        "dropout": 0.1,
    },
    "train": {
        STAGE: {
            "effective_batch_token_size": 40,
            "minibatch_token_size": 20,
            "learning_rate": 1.0e-3,
            "adam_betas": [0.9, 0.98],
            "adam_eps": 1.0e-9,
            "num_steps": 4,
            "label_smoothing": 0.1,
            "device": "cpu",
            "checkpoint_steps": 2,
            "warm_up_steps": 2,
            "validation_steps": 2,
            "validation_batches": 1,
            "compile_model": False,
        },
    },
    "data": {STAGE: {"ds": "techiaith/cardiff-university-tm-en-cy", "test_split_ratio": 0.25}},
    "tokenizer": {"vocab_size": 200, "pad_token_id": 2, "training_size": 1000},
    "locations": {"data_dir": "./data", "models_dir": "./models"},
}


@pytest.fixture
def config(tmp_path):
    config = copy.deepcopy(BASE_CONFIG)
    config["locations"]["models_dir"] = str(tmp_path / "models")
    config["locations"]["data_dir"] = str(tmp_path / "data")
    return config


@pytest.fixture
def raw_ds():
    def split(pairs):
        return Dataset.from_dict(
            {"text_en": [en for en, _ in pairs], "text_cy": [cy for _, cy in pairs]}
        )

    return DatasetDict(train=split(PAIRS * 4), test=split(PAIRS))


@pytest.fixture(scope="session")
def tok():
    ds = DatasetDict(
        train=Dataset.from_dict(
            {"text_en": [en for en, _ in PAIRS], "text_cy": [cy for _, cy in PAIRS]}
        )
    )
    return tokenizer.create_tokenizer(ds, BASE_CONFIG["tokenizer"])


@pytest.fixture
def ds_tokenized(raw_ds, tok, config):
    return tokenizer.tokenize_dataset(raw_ds, tok, config)


@pytest.fixture
def transformer(config, tok):
    torch.manual_seed(0)
    config["tokenizer"]["vocab_size"] = len(tok)
    return encoder_decoder.build_transformer(config)
