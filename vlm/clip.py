"""CLIP-style contrastive learning — §3."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHeads(nn.Module):
    """Linear projections into shared space with L2 normalization."""

    def __init__(self, d_image: int, d_text: int, d_proj: int = 256) -> None:
        super().__init__()
        self.image_proj = nn.Linear(d_image, d_proj, bias=False)
        self.text_proj = nn.Linear(d_text, d_proj, bias=False)

    def forward(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img = F.normalize(self.image_proj(image_embeds), dim=-1)
        txt = F.normalize(self.text_proj(text_embeds), dim=-1)
        return img, txt


def init_logit_scale() -> nn.Parameter:
    """CLIP-style learnable temperature, initialized to ln(1/0.07)."""
    return nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))


def clip_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE loss."""
    scale = logit_scale.exp()
    logits = image_embeds @ text_embeds.T * scale
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
