"""ViT variants with RoPE positional embeddings for §6 ablations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from basics.model import MLP, Block
from basics.rope import RoPE1D, RoPE2D
from basics.vit import PatchEmbeddings


class RoPEHead(nn.Module):
    """Self-attention head with RoPE applied to q and k."""

    def __init__(
        self,
        d_model: int,
        head_dim: int,
        block_size: int,
        rope: nn.Module,
        is_2d: bool = False,
        grid_size: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.is_2d = is_2d
        self.grid_size = grid_size
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)
        self.rope = rope
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.unsqueeze(1)  # (B, 1, T, head_dim) for rope
        k = k.unsqueeze(1)

        N = T - 1  # number of patch tokens (excluding CLS)

        if self.is_2d:
            gs = self.grid_size
            cls_q = q[:, :, :1, :]
            cls_k = k[:, :, :1, :]
            patch_q = q[:, :, 1:, :]
            patch_k = k[:, :, 1:, :]

            row_coords = torch.arange(gs, device=x.device).repeat_interleave(gs)
            col_coords = torch.arange(gs, device=x.device).repeat(gs)
            patch_q = self.rope(patch_q, row_coords, col_coords)
            patch_k = self.rope(patch_k, row_coords, col_coords)

            q = torch.cat([cls_q, patch_q], dim=2).squeeze(1)
            k = torch.cat([cls_k, patch_k], dim=2).squeeze(1)
        else:
            positions = torch.arange(T, device=x.device)
            q = self.rope(q, positions).squeeze(1)
            k = self.rope(k, positions).squeeze(1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return attn @ v


class RoPEMultiHeadAttention(nn.Module):
    """Multi-head attention using RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int,
        rope_module: nn.Module,
        is_2d: bool = False,
        grid_size: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList([
            RoPEHead(d_model, head_dim, block_size, rope_module, is_2d, grid_size, dropout)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.out_proj(out))


class RoPEBlock(nn.Module):
    """Transformer block using RoPE attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int,
        rope_module: nn.Module,
        is_2d: bool = False,
        grid_size: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadAttention(
            d_model, num_heads, block_size, rope_module, is_2d, grid_size, dropout
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViTRoPE1D(nn.Module):
    """ViT with 1D RoPE instead of learned positional embeddings."""

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
        self.num_patches = num_patches
        block_size = num_patches + 1
        head_dim = d_model // num_heads

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        rope = RoPE1D(head_dim, block_size + 1)
        self.blocks = nn.ModuleList([
            RoPEBlock(d_model, num_heads, block_size, rope, is_2d=False, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        if return_all_tokens:
            return x
        return x[:, 0]


class ViTRoPE2D(nn.Module):
    """ViT with 2D RoPE instead of learned positional embeddings."""

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
        self.num_patches = num_patches
        grid_size = img_size // patch_size
        self.grid_size = grid_size
        block_size = num_patches + 1
        head_dim = d_model // num_heads

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        rope = RoPE2D(head_dim, grid_size)
        self.blocks = nn.ModuleList([
            RoPEBlock(d_model, num_heads, block_size, rope, is_2d=True, grid_size=grid_size, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        if return_all_tokens:
            return x
        return x[:, 0]
