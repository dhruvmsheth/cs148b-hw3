"""Vision-Language Projector — §5."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionLanguageProjector(nn.Module):
    """2-layer MLP mapping image features to decoder embedding space."""

    def __init__(self, d_image: int, d_decoder: int, expansion: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_image, expansion * d_image)
        self.fc2 = nn.Linear(expansion * d_image, d_decoder)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        added_dim = False
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
            added_dim = True
        out = self.fc2(F.gelu(self.fc1(image_features)))
        if added_dim:
            pass  # keep the (B, 1, d_decoder) shape
        return out
