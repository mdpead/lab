import pytest
import torch

from src import generation

CPU = torch.device("cpu")


class ScriptedModel(torch.nn.Module):
    """Emits a fixed token sequence per row, ignoring the source."""

    def __init__(self, scripts, vocab_size):
        super().__init__()
        self.scripts = scripts
        self.vocab_size = vocab_size
        self.decode_calls = 0

    def encode(self, src, src_padding_mask):
        return torch.zeros(src.shape[0], src.shape[1], 1)

    def decode(self, src_enc, tgt, src_padding_mask, tgt_padding_mask):
        self.decode_calls += 1
        return tgt

    def output(self, tgt):
        step = tgt.shape[1] - 1
        logits = torch.zeros(tgt.shape[0], tgt.shape[1], self.vocab_size)
        for row, script in enumerate(self.scripts):
            logits[row, -1, script[min(step, len(script) - 1)]] = 1.0
        return logits


def ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def test_generate_stops_at_eos_and_decodes(tok):
    scripts = [ids(tok, "Bore da") + [tok.eos_token_id], ids(tok, "Diolch") + [tok.eos_token_id]]
    m = ScriptedModel(scripts, len(tok))

    texts = generation.generate_texts(m, tok, ["Good morning", "Thank you very much"], 32, CPU)

    assert texts == ["Bore da", "Diolch"]
    assert m.decode_calls == max(len(s) for s in scripts)


def test_finished_rows_stay_finished(tok):
    bore = ids(tok, "Bore")
    scripts = [[tok.eos_token_id] + bore * 5, bore * 3 + [tok.eos_token_id]]

    texts = generation.generate_texts(ScriptedModel(scripts, len(tok)), tok, ["a", "b"], 32, CPU)

    assert texts == ["", "Bore Bore Bore"]


def test_generation_capped_at_max_length(tok):
    m = ScriptedModel([ids(tok, "da")], len(tok))
    texts = generation.generate_texts(m, tok, ["Hi"], 8, CPU)
    assert m.decode_calls == 7
    assert texts == [" ".join(["da"] * 7)]


def test_rejoins_split_apostrophes(tok):
    script = ids(tok, "Mae ' r tywydd") + [tok.eos_token_id]
    texts = generation.generate_texts(ScriptedModel([script], len(tok)), tok, ["x"], 32, CPU)
    assert texts == ["Mae'r tywydd"]


def test_input_too_long_raises(tok):
    with pytest.raises(ValueError, match="too long"):
        generation.generate_texts(
            ScriptedModel([[1]], len(tok)), tok, ["Thank you very much"], 4, CPU
        )


def test_restores_training_mode(tok, transformer):
    transformer.train()
    generation.generate_texts(transformer, tok, ["Good morning"], 8, CPU)
    assert transformer.training

    transformer.eval()
    generation.generate_texts(transformer, tok, ["Good morning"], 8, CPU)
    assert not transformer.training
