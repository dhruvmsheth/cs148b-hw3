"""Vision Transformer — §2."""

from __future__ import annotations

import torch
import torch.nn as nn

from basics.model import Block


class PatchEmbeddings(nn.Module):
    """Conv2d-based patch embedding layer."""

    def __init__(self, img_size: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> (B, d_model, H/P, W/P) -> (B, num_patches, d_model)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT(nn.Module):
    """Vision Transformer with CLS token and learnable positional embeddings."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_embed = PatchEmbeddings(img_size, patch_size, d_model)
        num_patches = self.patch_embed.num_patches
        block_size = num_patches + 1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, block_size, is_decoder=False, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        if return_all_tokens:
            return x
        return x[:, 0]