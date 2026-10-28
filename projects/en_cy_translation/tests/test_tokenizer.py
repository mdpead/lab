import os

from src import tokenizer
from src.hf_wrapper import EnCyConfig


def test_special_token_ids_match_model_config(tok, config):
    hf_config = EnCyConfig()
    assert tok.bos_token_id == hf_config.bos_token_id == hf_config.decoder_start_token_id
    assert tok.eos_token_id == hf_config.eos_token_id
    assert tok.pad_token_id == hf_config.pad_token_id == config["tokenizer"]["pad_token_id"]


def test_encoding_is_wrapped_in_bos_and_eos(tok):
    ids = tok("Good morning")["input_ids"]
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.eos_token_id
    assert tok.unk_token_id not in ids


def test_decode_roundtrip(tok):
    for text in ["Good morning", "Diolch yn fawr iawn"]:
        assert tok.decode(tok(text)["input_ids"], skip_special_tokens=True) == text


def test_tokenize_dataset_adds_columns(ds_tokenized, tok):
    row = ds_tokenized["train"][0]
    assert row["text_en_tokenized"] == tok(row["text_en"])["input_ids"]
    assert row["text_cy_tokenized"] == tok(row["text_cy"])["input_ids"]
    assert row["en_token_length"] == len(row["text_en_tokenized"])
    assert row["cy_token_length"] == len(row["text_cy_tokenized"])
    assert row["length"] == max(row["en_token_length"], row["cy_token_length"])


def test_tokenize_dataset_filters_long_rows(raw_ds, tok, config):
    config["model"]["max_length"] = 5
    ds = tokenizer.tokenize_dataset(raw_ds, tok, config)
    assert 0 < len(ds["train"]) < len(raw_ds["train"])
    assert max(list(ds["train"]["en_token_length"]) + list(ds["train"]["cy_token_length"])) <= 5


def test_get_tokenizer_saves_then_reuses(raw_ds, config, monkeypatch):
    first = tokenizer.get_tokenizer(raw_ds, config)
    assert os.path.isdir(f"{config['locations']['models_dir']}/pytest/tokenizer")

    def fail(*args, **kwargs):
        raise AssertionError("tokenizer should be loaded, not retrained")

    monkeypatch.setattr(tokenizer, "create_tokenizer", fail)
    second = tokenizer.get_tokenizer(raw_ds, config)
    assert second.get_vocab() == first.get_vocab()
