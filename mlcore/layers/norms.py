from dataclasses import dataclass

import torch
from torch import Tensor, nn

from mlcore.config import Registry


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x_prec = x.to(torch.float32)
        ms = (x_prec**2).mean(-1, keepdim=True)
        x_scaled = x_prec * torch.rsqrt(ms + self.eps)
        out = x_scaled * self.alpha
        out = out.to(x.dtype)
        return out


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        # eps on the std, not the variance, as the encoder-decoder was trained with
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


NORMS = Registry("norm")


@dataclass(frozen=True, kw_only=True)
class NormConfig:
    """A normalisation layer over the model dimension."""

    eps: float = 1e-6

    def build(self, d_model: int) -> nn.Module:
        raise NotImplementedError


@NORMS.register("rms")
@dataclass(frozen=True, kw_only=True)
class RMSNormConfig(NormConfig):
    def build(self, d_model: int) -> nn.Module:
        return RMSNorm(d_model, self.eps)


@NORMS.register("layer")
@dataclass(frozen=True, kw_only=True)
class LayerNormConfig(NormConfig):
    def build(self, d_model: int) -> nn.Module:
        return LayerNorm(d_model, self.eps)
