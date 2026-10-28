import torch
from torch import Tensor, nn


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_h: int, max_length: int) -> None:
        super().__init__()
        cos_thetas, sin_thetas = self.create_ang_trigs(d_h, max_length)
        self.register_buffer("cos_thetas", cos_thetas, persistent=False)
        self.register_buffer("sin_thetas", sin_thetas, persistent=False)

    @staticmethod
    def create_ang_trigs(d_h: int, max_length: int) -> tuple[Tensor, Tensor]:
        base = 10000

        ws = []
        for i in range(0, d_h // 2):
            w_i = base ** (-2 * i / d_h)
            ws.append(w_i)
        ws = torch.tensor(ws, dtype=torch.float32)

        # Calculate the rotation angle for each pair
        positions = torch.arange(max_length)
        thetas = positions[:, None] * ws[None, :]  # (seq, d_h/2)

        # Calculate the trig components needed
        cos_thetas = torch.cos(thetas)
        sin_thetas = torch.sin(thetas)

        return cos_thetas, sin_thetas

    @staticmethod
    def rotate(x: Tensor, cos_thetas: Tensor, sin_thetas: Tensor) -> Tensor:
        cos_thetas = cos_thetas.to(x.dtype)
        sin_thetas = sin_thetas.to(x.dtype)

        # Apply the rotations to each pair
        x1 = x[..., 0::2]  # (b, h, seq, d_h/2)
        x2 = x[..., 1::2]  # (b, h, seq, d_h/2)

        out1 = x1 * cos_thetas - x2 * sin_thetas  # (b, h, seq, d_h/2)
        out2 = x1 * sin_thetas + x2 * cos_thetas  # (b, h, seq, d_h/2)

        # Interleave back together
        out = torch.stack([out1, out2], dim=-1).flatten(-2)  # (b, h, seq, d_h)

        return out

    def forward(self, Q: Tensor, K: Tensor, offset: int) -> tuple[Tensor, Tensor]:

        seq_length = Q.shape[-2]

        if offset + seq_length > self.cos_thetas.shape[0]:
            raise ValueError(
                f"offset {offset} + seq_length {seq_length} exceeds "
                f"max_length {self.cos_thetas.shape[0]}"
            )

        cos_thetas = self.cos_thetas[offset : offset + seq_length]
        sin_thetas = self.sin_thetas[offset : offset + seq_length]

        Q_rot = self.rotate(Q, cos_thetas, sin_thetas)
        K_rot = self.rotate(K, cos_thetas, sin_thetas)

        return Q_rot, K_rot


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sin/cos encodings added to the embeddings, as in the original Transformer."""

    def __init__(self, d_model: int, max_length: int) -> None:
        super().__init__()
        # Persistent, so it is part of saved checkpoints
        self.register_buffer("PE", self.create_pe(d_model, max_length))

    @staticmethod
    def create_pe(d_model: int, max_length: int) -> Tensor:
        PE = torch.zeros((max_length, d_model))
        positions = torch.arange(0, max_length)
        dims = torch.arange(0, d_model // 2)

        evens = torch.sin(positions.unsqueeze(-1) / (10000 ** (2 * dims / d_model)).unsqueeze(0))
        odds = torch.cos(positions.unsqueeze(-1) / (10000 ** (2 * dims / d_model)).unsqueeze(0))

        PE[:, 0::2] = evens
        PE[:, 1::2] = odds

        return PE

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.PE[: x.shape[1], :].unsqueeze(0)
        return x
