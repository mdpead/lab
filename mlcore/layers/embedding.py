import math

import torch
from torch import Tensor, nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.E = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        x = self.E[x] * math.sqrt(self.d_model)
        return x
