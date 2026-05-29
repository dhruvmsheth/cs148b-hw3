"""LoRA adapters — §4."""

from __future__ import annotations

import torch
import torch.nn as nn

from basics.model import Head


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping an existing nn.Linear layer."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.base_layer = base_layer

        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        d_in = base_layer.in_features
        d_out = base_layer.out_features
        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.scaling * (x @ self.A.T @ self.B.T)


def apply_lora_to_attention(model: nn.Module, rank: int, alpha: float) -> nn.Module:
    """Replace q_proj and v_proj in every Head with LoRA-wrapped versions."""
    for p in model.parameters():
        p.requires_grad_(False)

    for name, module in model.named_modules():
        if isinstance(module, Head):
            module.q_proj = LoRALinear(module.q_proj, rank, alpha)
            module.v_proj = LoRALinear(module.v_proj, rank, alpha)

    return model
