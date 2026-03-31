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

from .tnet import FeatureTNet, InputTNet


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


class PointNet(nn.Module):
    """
    Global PointNet backbone on batched XYZ clouds.

    For each batch item, with points X of shape [N, 3]:

    - A = input_tnet(X)  → [3, 3]; X ← X @ A
    - F = shared_mlp1(X) → [N, D1]
    - B = feature_tnet(F) → [D1, D1]; F ← F @ B
    - F = shared_mlp2(F) → [N, D2]
    - g = max(F, dim=points) → [D2]

    Args:
        mlp1_dims: Channel widths for the first per-point MLP (input 3).
        mlp2_dims: Channel widths for the second per-point MLP (input mlp1_dims[-1]).
        use_input_transform: If False, skip input T-Net (no X @ A).
        use_feature_transform: If False, skip feature T-Net (no F @ B).
    """

    def __init__(
        self,
        mlp1_dims: list[int] | None = None,
        mlp2_dims: list[int] | None = None,
        *,
        use_input_transform: bool = True,
        use_feature_transform: bool = True,
    ) -> None:
        super().__init__()
        m1 = list(mlp1_dims) if mlp1_dims is not None else [64, 64]
        m2 = list(mlp2_dims) if mlp2_dims is not None else [64, 128, 1024]
        if not m1:
            raise ValueError("mlp1_dims must be non-empty")
        if not m2:
            raise ValueError("mlp2_dims must be non-empty")

        self.mlp1_dims = m1
        self.mlp2_dims = m2
        self.use_input_transform = use_input_transform
        self.use_feature_transform = use_feature_transform

        d1 = m1[-1]

        self.input_tnet = InputTNet(3, 3) if use_input_transform else None
        self.mlp1 = _make_h_mlp(3, m1)

        self.feature_tnet = FeatureTNet(d1, d1) if use_feature_transform else None
        self.mlp2 = _make_h_mlp(d1, m2)

    @property
    def global_dim(self) -> int:
        return self.mlp2_dims[-1]

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) point coordinates.
            mask: Optional (B, N) bool, True for valid points (padding ignored in max pool).

        Returns:
            Global feature g of shape (B, global_dim).
        """
        if x.dim() != 3:
            raise ValueError(f"x must be (B, N, 3), got {tuple(x.shape)}")
        if x.size(-1) != 3:
            raise ValueError(f"expected last dim 3 for XYZ, got {x.size(-1)}")

        b, n, _ = x.shape
        if mask is not None and mask.shape != (b, n):
            raise ValueError(
                f"mask must be (B, N) = {(b, n)}, got {tuple(mask.shape)}"
            )

        if self.input_tnet is not None:
            a = self.input_tnet(x.transpose(1, 2))
            x = torch.bmm(x, a)

        x = self.mlp1(x)

        if self.feature_tnet is not None:
            bmat = self.feature_tnet(x.transpose(1, 2))
            x = torch.bmm(x, bmat)

        x = self.mlp2(x)

        if mask is not None:
            neg = torch.finfo(x.dtype).min / 2
            x = x.masked_fill(~mask.unsqueeze(-1), neg)

        return torch.max(x, dim=1).values


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
        if x.dim() != 4:
            raise ValueError(f"x must be (B, S, K, C), got {x.shape}")
        b, s, k, c_in = x.shape
        if c_in != self.in_channels:
            raise ValueError(
                f"expected C_in={self.in_channels}, got {c_in}"
            )
        if mask is not None:
            if mask.shape != (b, s, k):
                raise ValueError(
                    f"mask must be (B, S, K) = {(b, s, k)}, got {mask.shape}"
                )

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
