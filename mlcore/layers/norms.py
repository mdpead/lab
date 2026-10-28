import torch
from torch import Tensor, nn


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
