import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lab.config import Registry

FEEDFORWARDS = Registry("feedforward", provided=("d_model",))


@FEEDFORWARDS.register("relu")
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = False) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


@FEEDFORWARDS.register("swiglu")
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        g = F.silu(self.gate(x))
        u = self.up(x)
        u_g = g * u
        o = self.down(u_g)
        return o


@FEEDFORWARDS.register("moe")
class MixtureOfExpertsFeedForward(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, num_experts: int, top_k: int, capacity_factor: float
    ) -> None:
        super().__init__()
        self.check_config(num_experts=num_experts, top_k=top_k)
        self.num_experts = num_experts
        self.top_k = top_k
        # Rows per expert, as a multiple of its balanced share. Above 1.0 absorbs router
        # imbalance, at the cost of padded compute.
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUFeedForward(d_model, d_ff) for _ in range(0, num_experts)]
        )

    @staticmethod
    def check_config(num_experts: int, top_k: int, **_: object) -> None:
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k ({top_k}) must be between 1 and num_experts ({num_experts})")

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:

        n, d = x.shape[0] * x.shape[1], x.shape[-1]

        x_flat = x.reshape(-1, x.shape[-1])  # (batch * seq = n, d)

        # Pad tokens aren't data: without this they take expert seats and skew the balance loss.
        # Capacity still derives from the padded n.
        real = (
            x.new_ones((n, 1), dtype=torch.bool)
            if padding_mask is None
            else padding_mask.reshape(n, 1)
        )
        router_logits = self.router(x_flat)
        route_l, route_ind = torch.topk(router_logits, self.top_k, dim=-1)
        route_p = F.softmax(route_l, dim=-1)  # (n, k)
        full_p = F.softmax(router_logits, dim=-1, dtype=torch.float32)  # (n, e)

        # Assign seats to each token
        token_expert_oh = F.one_hot(route_ind, self.num_experts) * real.unsqueeze(-1)  # (n, k, e)
        rank_km = (
            token_expert_oh.permute(1, 0, 2).reshape(route_ind.shape[0] * self.top_k, -1).cumsum(0)
            - 1
        )  # (n * k, e)
        token_expert_seat = rank_km.reshape(
            self.top_k, route_ind.shape[0], self.num_experts
        ).permute(1, 0, 2)  # (n, k, e)
        token_choice_seat = token_expert_seat.gather(-1, route_ind.unsqueeze(-1)).squeeze(
            -1
        )  # (n, k)

        # Capacity only drops tokens in training, where it bounds padded compute.
        # At inference n can be one decode step (zero seats), so n seats never overflow.
        if self.training:
            capacity = int(self.capacity_factor * n * self.top_k / self.num_experts)
        else:
            capacity = n
        # A pad never keeps its seat, so one condition covers overflow and padding
        keep = (token_choice_seat < capacity) & real
        gate = route_p * keep
        seat_safe = torch.where(keep, token_choice_seat, capacity)  # (n, k)

        # Dispatch
        token_idx = torch.arange(n, device=x.device).unsqueeze(1).expand(n, self.top_k)  # (n, k)
        t_flat, e_flat, s_flat = (
            token_idx.reshape(-1),
            route_ind.reshape(-1),
            seat_safe.reshape(-1),
        )  # (n * k,)

        buf = x_flat.new_zeros(self.num_experts, capacity + 1, d)
        buf[e_flat, s_flat] = x_flat[t_flat]  # (e, c + 1, d)

        # Fixed (capacity + 1, d) slice per expert. Row `capacity` is an overflow bin
        # for dropped assignments, which carry a zero gate.
        y = torch.stack(
            [self.experts[i](buf[i]) for i in range(0, self.num_experts)], dim=0
        )  # (e, c + 1, d)

        # index_add_ combines each token's top_k results.
        # Cast to the residual dtype: experts return fp16 under autocast.
        out = x_flat.new_zeros(n, d)
        weighted = y[e_flat, s_flat] * gate.reshape(-1, 1).to(y.dtype)
        out.index_add_(0, t_flat, weighted.to(out.dtype))

        # Load balance is measured on the routing decision, not on what survived capacity
        counted = token_expert_oh.float()  # already zero on pad rows
        expert_fracs = counted.sum(dim=(0, 1)) / counted.sum().clamp(min=1.0)  # (e), sums to 1
        expert_probs = (full_p * real).sum(dim=0) / real.sum().clamp(min=1)  # (e)

        loss_aux = self.num_experts * (expert_fracs * expert_probs).sum()

        return out.reshape(x.shape), loss_aux
