"""
PointNet variants:

- PointNet: global feature extraction on XYZ point clouds (input + feature T-Net,
  two shared MLPs, max pooling over points).
- PointNetLocal: mini-PointNet on local neighborhoods for PointNet++ (shared MLP,
  max over neighbors, optional γ).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _make_h_mlp(in_channels: int, mlp_dims: list[int]) -> nn.Sequential:
    """Shared MLP: same Linear map on each point, input (…, C_in) → (…, C_out)."""
    layers: list[nn.Module] = []
    c = in_channels
    for d in mlp_dims:
        layers.append(nn.Linear(c, d))
        layers.append(nn.ReLU(inplace=True))
        c = d
    return nn.Sequential(*layers)


def _make_gamma_mlp(in_channels: int, dims: list[int]) -> nn.Sequential | None:
    """MLP after max pooling to output the global feature."""
    if not dims:
        return None
    layers: list[nn.Module] = []
    c = in_channels
    for i, d in enumerate(dims):
        layers.append(nn.Linear(c, d))
        if i < len(dims) - 1:
            layers.append(nn.ReLU(inplace=True))
        c = d
    return nn.Sequential(*layers)

class PointNetLocal(nn.Module):
    """
    PointNet on a local neighborhood: γ(max_i h(x_i)).

    Input shape (B, S, K, C_in) — batch, centroids, neighbors, channels.
    """

    def __init__(
        self,
        in_channels: int,
        mlp_dims: list[int],
        post_mlp_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if not mlp_dims:
            raise ValueError("mlp_dims must be non-empty")
        self.in_channels = in_channels
        self.mlp_dims = mlp_dims
        self.post_mlp_dims = post_mlp_dims

        self.h = _make_h_mlp(in_channels, mlp_dims)
        self.gamma = _make_gamma_mlp(mlp_dims[-1], post_mlp_dims or [])

    @property
    def out_channels(self) -> int:
        if self.post_mlp_dims:
            return self.post_mlp_dims[-1]
        return self.mlp_dims[-1]

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        b, s, k, c_in = x.shape

        x = x.reshape(b * s, k, c_in)
        x = self.h(x)
        if mask is not None:
            neg = torch.finfo(x.dtype).min / 2
            m = mask.reshape(b * s, k).unsqueeze(-1)
            x = x.masked_fill(~m, neg)
        x = torch.max(x, dim=1).values
        x = x.view(b, s, -1)

        if self.gamma is not None:
            x = self.gamma(x)
        return x
