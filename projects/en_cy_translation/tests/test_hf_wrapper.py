import os

import pytest
import torch

from src import generation, train, utils
from src.hf_wrapper import EnCyConfig, EnCyForTranslation


def make_hf_model(config, transformer):
    model_cfg = config["model"]
    hf_config = EnCyConfig(
        vocab_size=config["tokenizer"]["vocab_size"],
        d_model=model_cfg["d_model"],
        num_heads=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"],
        num_enc_layers=model_cfg["num_enc_layers"],
        num_dec_layers=model_cfg["num_dec_layers"],
        max_length=model_cfg["max_length"],
        dropout=model_cfg["dropout"],
    )
    model = EnCyForTranslation(hf_config)
    model.transformer.load_state_dict(transformer.state_dict())
    return model.eval()


def test_forward_matches_transformer(config, transformer):
    transformer.eval()
    hf_model = make_hf_model(config, transformer)
    src = torch.randint(5, 50, (2, 6))
    tgt = torch.randint(5, 50, (2, 4))
    src_mask = torch.tensor([[1] * 6, [1] * 4 + [0] * 2])

    out = hf_model(input_ids=src, attention_mask=src_mask, decoder_input_ids=tgt)
    expected = transformer(src, tgt, src_mask.bool(), torch.ones(2, 4, dtype=torch.bool))

    assert torch.allclose(out.logits, expected, atol=1e-5)
    assert out.loss is None


def test_missing_attention_mask_attends_to_all_tokens(config, transformer):
    hf_model = make_hf_model(config, transformer)
    src = torch.randint(5, 50, (2, 6))
    tgt = torch.randint(5, 50, (2, 4))

    expected = hf_model(
        input_ids=src, attention_mask=torch.ones_like(src), decoder_input_ids=tgt
    ).logits
    from_ids = hf_model(input_ids=src, decoder_input_ids=tgt).logits
    from_encoder = hf_model(
        encoder_outputs=hf_model.get_encoder()(src), decoder_input_ids=tgt
    ).logits

    assert torch.allclose(from_ids, expected)
    assert torch.allclose(from_encoder, expected)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(input_ids=torch.randint(5, 50, (1, 6))), "decoder_input_ids"),
        (dict(decoder_input_ids=torch.randint(5, 50, (1, 4))), "input_ids or encoder_outputs"),
    ],
)
def test_forward_rejects_missing_inputs(config, transformer, kwargs, message):
    hf_model = make_hf_model(config, transformer)
    with pytest.raises(ValueError, match=message):
        hf_model(**kwargs)


def test_loss_ignores_padding(config, transformer):
    hf_model = make_hf_model(config, transformer)
    src = torch.randint(5, 50, (1, 6))
    tgt = torch.randint(5, 50, (1, 4))
    labels = torch.randint(5, 50, (1, 4))
    labels_padded = labels.clone()
    labels_padded[0, 2:] = hf_model.config.pad_token_id

    out = hf_model(
        input_ids=src,
        attention_mask=torch.ones_like(src),
        decoder_input_ids=tgt,
        labels=labels_padded,
    )
    expected = torch.nn.functional.cross_entropy(out.logits[0, :2], labels[0, :2])

    assert torch.allclose(out.loss, expected, atol=1e-5)


def test_generate_matches_greedy_generation(config, transformer, tok):
    hf_model = make_hf_model(config, transformer)
    texts = ["Good morning", "Where is the station?"]

    expected = generation.generate_texts(transformer, tok, texts, 12, torch.device("cpu"))
    inputs = tok(texts, return_tensors="pt", padding=True)
    output_ids = hf_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=12,
        do_sample=False,
        num_beams=1,
    )
    actual = tok.batch_decode(output_ids, skip_special_tokens=True)

    assert actual == expected


def assert_roundtrip(tmp_path, hf_model):
    hf_model.save_pretrained(tmp_path)
    loaded = EnCyForTranslation.from_pretrained(tmp_path).eval()

    src = torch.randint(5, 50, (1, 6))
    tgt = torch.randint(5, 50, (1, 3))
    kwargs = dict(input_ids=src, attention_mask=torch.ones_like(src), decoder_input_ids=tgt)
    assert loaded.config.d_model == hf_model.config.d_model
    assert torch.allclose(loaded(**kwargs).logits, hf_model(**kwargs).logits)


def test_save_and_load_pretrained_default_max_length(tmp_path, config, transformer):
    config["model"]["max_length"] = EnCyConfig().max_length
    model_cfg = config["model"]
    transformer.positional_encoder = type(transformer.positional_encoder)(
        model_cfg["d_model"], model_cfg["max_length"]
    )
    assert_roundtrip(tmp_path, make_hf_model(config, transformer))


@pytest.mark.xfail(
    strict=True,
    reason="save_pretrained moves a non-default max_length into generation_config.json, "
    "so it reloads as None",
)
def test_save_and_load_pretrained_custom_max_length(tmp_path, config, transformer):
    assert_roundtrip(tmp_path, make_hf_model(config, transformer))


@pytest.mark.xfail(
    strict=True,
    reason="README usage: the tokenizer returns token_type_ids, which generate() rejects",
)
def test_generate_accepts_tokenizer_output(config, transformer, tok):
    hf_model = make_hf_model(config, transformer)
    inputs = tok("Good morning", return_tensors="pt")
    hf_model.generate(**inputs, max_length=12)


def test_from_run_loads_latest_checkpoint(config, transformer):
    run_path = utils.init_run(config)
    run = train.create_run(transformer, config)
    args = run["optimiser"], run["lr_scheduler"], run["scaler"], run_path
    train.save_checkpoint(torch.nn.Linear(1, 1), *args, 1, [])
    train.save_checkpoint(transformer, *args, 10, [])
    train.save_checkpoint(torch.nn.Linear(1, 1), *args, 9, [])
    assert sorted(os.listdir(f"{run_path}/checkpoints")) == ["1.pt", "10.pt", "9.pt"]

    hf_model = EnCyForTranslation.from_run(run_path)

    for key, value in transformer.state_dict().items():
        assert torch.equal(hf_model.transformer.state_dict()[key], value)
