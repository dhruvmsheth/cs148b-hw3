"""Rotary Position Embeddings — §6."""

from __future__ import annotations

import torch
import torch.nn as nn


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotation to paired dimensions. x: (..., d), cos/sin: (..., d//2)."""
    d = x.shape[-1]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    # Interleave back
    out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
    return out


class RoPE1D(nn.Module):
    """1D Rotary Position Embedding."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: (B, num_heads, T, head_dim), positions: (T,)
        cos = self.cos_cached[positions]  # (T, head_dim//2)
        sin = self.sin_cached[positions]  # (T, head_dim//2)
        # Broadcast to (1, 1, T, head_dim//2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return _apply_rope(x, cos, sin)


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding for image patches."""

    def __init__(self, head_dim: int, grid_size: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 4 == 0
        self.head_dim = head_dim
        self.grid_size = grid_size
        self.base = base

        half_dim = head_dim // 2
        inv_freq = base ** (-torch.arange(0, half_dim, 2).float() / half_dim)
        t = torch.arange(grid_size).float()
        freqs = torch.outer(t, inv_freq)  # (grid_size, head_dim // 4)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, num_heads, T, head_dim)
        half = self.head_dim // 2
        x_first = x[..., :half]
        x_second = x[..., half:]

        cos_x = self.cos_cached[x_coords].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim//4)
        sin_x = self.sin_cached[x_coords].unsqueeze(0).unsqueeze(0)
        cos_y = self.cos_cached[y_coords].unsqueeze(0).unsqueeze(0)
        sin_y = self.sin_cached[y_coords].unsqueeze(0).unsqueeze(0)

        out_first = _apply_rope(x_first, cos_x, sin_x)
        out_second = _apply_rope(x_second, cos_y, sin_y)
        return torch.cat([out_first, out_second], dim=-1)
