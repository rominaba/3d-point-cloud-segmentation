"""
PointNet++ Feature Propagation (upsampling) layer.

Given coarse features at xyz_coarse and optionally fine skip features at
xyz_fine, this layer interpolates coarse features onto fine points using
inverse-distance weights, then applies a shared MLP on the concatenated vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _make_mlp(in_dim: int, dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    c = in_dim
    for d in dims:
        layers.append(nn.Linear(c, d))
        layers.append(nn.ReLU(inplace=True))
        c = d
    return nn.Sequential(*layers)


class FeaturePropagation(nn.Module):
    """
    Feature propagation from a coarse point set to a finer one.

    Args:
        k: number of nearest neighbors used for interpolation.
        mlp_dims: list of output dims; last dim is the output feature dim.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        k: int = 3,
        mlp_dims: list[int],
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not mlp_dims:
            raise ValueError("mlp_dims must be non-empty")
        self.k = k
        self.in_channels = in_channels
        self.mlp_dims = list(mlp_dims)
        self.mlp = _make_mlp(in_channels, self.mlp_dims)

    def forward(
        self,
        xyz_coarse: torch.Tensor,
        feats_coarse: torch.Tensor,
        xyz_fine: torch.Tensor,
        feats_fine_skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            xyz_coarse: Lower resolution point cloud (B, Nc, 3)
            feats_coarse: Lower resolution point features (B, Nc, Cc)
            xyz_fine: Higher resolution point cloud (B, Nf, 3)
            feats_fine_skip: optional higher resolution point features (B, Nf, Cf)

        Returns:
            (B, Nf, C_out)
        """
        

        b, nc, _ = xyz_coarse.shape
        nf = xyz_fine.size(1)
        # Pairwise distances for kNN interpolation.
        dist = torch.cdist(xyz_fine, xyz_coarse)  # (B, Nf, Nc)
        k_use = min(self.k, nc)
        dist_sel, idx = dist.topk(k_use, dim=-1, largest=False)  # (B, Nf, k)

        # Inverse-distance weights (normalized).
        eps = 1e-10
        w = 1.0 / (dist_sel + eps) # TODO Try ^2
        w = w / w.sum(dim=-1, keepdim=True)

        # Gather coarse features: (B, Nf, k, Cc)
        batch = torch.arange(b, device=xyz_coarse.device).view(b, 1, 1).expand(b, nf, k_use)
        feats_sel = feats_coarse[batch, idx]  # advanced indexing
        feats_interp = (feats_sel * w.unsqueeze(-1)).sum(dim=2)  # (B, Nf, Cc)

   
        feats_cat = (
            torch.cat([feats_interp, feats_fine_skip], dim=-1)
            if feats_fine_skip is not None
            else feats_interp
        )
        

        return self.mlp(feats_cat)

