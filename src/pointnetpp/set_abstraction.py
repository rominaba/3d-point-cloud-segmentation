"""
PointNet++ Set Abstraction layers for PointNet++-style feature learning.

A simple downsampling + local feature extraction layer built
on top of the MultiResolutionPointNetLocal module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .multiresolution import MultiResolutionPointNetLocal
from .sampling import farthest_point_sample, index_points


class SetAbstractionMRG(nn.Module):
    """
    Downsample points with FPS, then extract features at the centroids using MRG.

    Input point tensor has shape (B, N, C) where the first 3 channels are XYZ.
    Output:
      - new_xyz: (B, S, 3)
      - new_features: (B, S, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        *,
        npoint: int,
        coarse_npoint: int,
        fine_radius: float,
        fine_max_neighbors: int,
        fine_mlp_dims: list[int],
        coarse_radius: float,
        coarse_max_neighbors: int,
        coarse_mlp_dims: list[int],
        fused_dim: int | None = None,
        deterministic_start: bool = False,
    ) -> None:
        super().__init__()
        if npoint < 1:
            raise ValueError(f"npoint must be >= 1, got {npoint}")
        if coarse_npoint < 1:
            raise ValueError(f"coarse_npoint must be >= 1, got {coarse_npoint}")
        
        self.in_channels = in_channels
        # Number of centroids for the next layer.
        self.npoint = npoint
        # Number of centroids for the coarse MRG branch.
        self.coarse_npoint = coarse_npoint
        self.deterministic_start = deterministic_start

        self.mrg = MultiResolutionPointNetLocal(
            in_channels=in_channels,
            fine_radius=fine_radius,
            fine_max_neighbors=fine_max_neighbors,
            fine_mlp_dims=fine_mlp_dims,
            coarse_radius=coarse_radius,
            coarse_max_neighbors=coarse_max_neighbors,
            coarse_mlp_dims=coarse_mlp_dims,
            npoint=npoint,
            deterministic_start=deterministic_start,
            fused_dim=fused_dim,
        )

    @property
    def out_channels(self) -> int:
        return self.mrg.out_channels

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, C) point tensor, C >= 3 and x[..., :3] are XYZ.

        Returns:
            new_xyz: (B, S, 3)
            new_features: (B, S, out_channels)
        """
        b, n, c = x.shape
        # Centroids (fine resolution) for the next layer.
        fps_idx = farthest_point_sample(
            x, self.npoint, deterministic_start=self.deterministic_start
        )
        # Gather the fine point cloud and features
        new_xyz = index_points(x[..., :3], fps_idx)

        # Coarse point set used by the coarse MRG branch.
        coarse_idx = farthest_point_sample(
            x, self.coarse_npoint, deterministic_start=self.deterministic_start
        )
        # Gather the coarse point cloud and features
        coarse_xyz_and_feats = index_points(x, coarse_idx)

        # Compute the fused features
        fused, _, _w = self.mrg(
            fine_xyz=x,
            coarse_xyz=coarse_xyz_and_feats,
            centroid_xyz=new_xyz,
        )
        return new_xyz, fused

