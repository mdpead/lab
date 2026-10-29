from typing import Any

import torch
import torch.nn as nn
from lab import runs, training
from lab.models import parse_model
from lab.models.encoder_decoder import ModelConfig, Transformer
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


class EnCyConfig(PretrainedConfig):
    model_type = "en_cy_transformer"

    def __init__(
        self,
        vocab_size: int = 16000,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_enc_layers: int = 6,
        num_dec_layers: int = 6,
        max_length: int = 256,
        dropout: float = 0.1,
        pad_token_id: int = 2,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        decoder_start_token_id: int = 0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("is_encoder_decoder", True)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.num_hidden_layers = num_dec_layers
        self.max_length = max_length
        self.dropout = dropout


# EnCyConfig's flat fields, which are what the Hub stores
_FLAT_FIELDS = ("d_model", "num_heads", "d_ff", "num_enc_layers", "num_dec_layers", "max_length")


def _model_config(flat: dict[str, Any]) -> ModelConfig:
    """The architecture the Hub format describes: the original one, with these sizes."""
    model = {
        "type": "encoder_decoder",
        "d_model": flat["d_model"],
        "num_enc_layers": flat["num_enc_layers"],
        "num_dec_layers": flat["num_dec_layers"],
        "max_length": flat["max_length"],
        "dropout": flat["dropout"],
        "norm": {"type": "layer"},
        "position": {"type": "sinusoidal"},
        "attention": {"type": "mha", "num_heads": flat["num_heads"]},
        "feedforward": {"type": "relu", "d_ff": flat["d_ff"], "bias": True},
    }
    return parse_model(model, "encoder_decoder")


def _flat_fields(model_config: ModelConfig) -> dict[str, Any]:
    """EnCyConfig's fields for a model config, which must be the architecture they imply."""
    flat = {
        "d_model": model_config.d_model,
        "num_heads": model_config.attention["num_heads"],
        "d_ff": model_config.feedforward["d_ff"],
        "num_enc_layers": model_config.num_enc_layers,
        "num_dec_layers": model_config.num_dec_layers,
        "max_length": model_config.max_length,
        "dropout": model_config.dropout,
    }
    if _model_config(flat) != model_config:
        raise ValueError(
            "EnCyForTranslation describes only the original architecture (LayerNorm, "
            "sinusoidal positions, multi-head attention, ReLU feed-forward with biases); "
            f"this run's model is {model_config}"
        )
    return flat


class _EncoderWrapper(torch.nn.Module):
    """Wrapper around transformer.encode() returning BaseModelOutput."""

    def __init__(self, transformer: Transformer) -> None:
        super().__init__()
        self._transformer = transformer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutput:
        if attention_mask is not None:
            mask = attention_mask.bool()
        else:
            mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
        hidden = self._transformer.encode(input_ids, mask)
        return BaseModelOutput(last_hidden_state=hidden)


class EnCyForTranslation(PreTrainedModel, GenerationMixin):
    config_class = EnCyConfig

    def __init__(self, config: EnCyConfig) -> None:
        super().__init__(config)
        flat = {name: getattr(config, name) for name in (*_FLAT_FIELDS, "dropout")}
        self.transformer = Transformer(_model_config(flat), config.vocab_size)
        self.post_init()

    def get_encoder(self) -> _EncoderWrapper:
        return _EncoderWrapper(self.transformer)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_outputs: BaseModelOutput | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Seq2SeqLMOutput:
        if decoder_input_ids is None:
            raise ValueError("decoder_input_ids must be provided")

        if encoder_outputs is None:
            if input_ids is None:
                raise ValueError("Either input_ids or encoder_outputs must be provided")
            src_shape, src_device = input_ids.shape, input_ids.device
        else:
            src_shape = encoder_outputs.last_hidden_state.shape[:2]
            src_device = encoder_outputs.last_hidden_state.device

        if attention_mask is not None:
            src_mask = attention_mask.bool()
        else:
            src_mask = torch.ones(src_shape, dtype=torch.bool, device=src_device)

        if decoder_attention_mask is not None:
            tgt_mask = decoder_attention_mask.bool()
        else:
            tgt_mask = torch.ones(
                decoder_input_ids.shape, dtype=torch.bool, device=decoder_input_ids.device
            )

        if encoder_outputs is None:
            enc = self.transformer.encode(input_ids, src_mask)
        else:
            enc = encoder_outputs.last_hidden_state

        tgt_dec = self.transformer.decode(enc, decoder_input_ids, src_mask, tgt_mask)
        logits = self.transformer.output(tgt_dec)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            encoder_last_hidden_state=enc,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids: torch.Tensor,
        past_key_values: Any = None,
        attention_mask: torch.Tensor | None = None,
        encoder_outputs: BaseModelOutput | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
        }

    @classmethod
    def from_run(cls, stage_path: str) -> "EnCyForTranslation":
        """Load the latest checkpoint of a training stage, e.g. models/base/translation."""
        config_dict = runs.load_config(stage_path)
        model_config = parse_model(config_dict["model"], "encoder_decoder")
        hf_config = EnCyConfig(
            vocab_size=config_dict["tokenizer"]["vocab_size"], **_flat_fields(model_config)
        )
        model = cls(hf_config)
        steps = training.checkpoint_steps_on_disk(f"{stage_path}/checkpoints")
        if not steps:
            raise FileNotFoundError(f"No checkpoints found in {stage_path}")
        checkpoint = training.load_checkpoint(stage_path, steps[-1], torch.device("cpu"))
        model.transformer.load_state_dict(checkpoint["model_state_dict"])
        return model
