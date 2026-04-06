"""
PointNet++ Set Abstraction layers for PointNet++-style feature learning.

Downsampling (FPS) plus local feature extraction using either
Multi-Resolution Grouping (MRG) or Multi-Scale Grouping (MSG).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .multiresolution import MultiResolutionPointNetLocal
from .multiscale import MultiScalePointNetLocal
from .sampling import farthest_point_sample, index_points

AggregationMode = Literal["multiresolution", "multiscale"]


class SetAbstraction(nn.Module):
    """
    Downsample points with FPS, then extract features at centroids.

    Use aggregation="multiresolution" for MRG (fine + coarse point sets)
    or aggregation="multiscale" for MSG (multiple radii on one point set).

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
        aggregation: AggregationMode = "multiresolution",
        deterministic_start: bool = False,
        # multiresolution (MRG)
        coarse_npoint: int | None = None,
        fine_radius: float = 0.2,
        fine_max_neighbors: int = 16,
        fine_mlp_dims: list[int] | None = None,
        coarse_radius: float = 0.4,
        coarse_max_neighbors: int = 32,
        coarse_mlp_dims: list[int] | None = None,
        fused_dim: int | None = None,
        # multiscale (MSG)
        msg_radii: list[float] | None = None,
        msg_max_neighbors: list[int] | None = None,
        msg_mlp_dims_per_scale: list[list[int]] | None = None,
    ) -> None:
        super().__init__()
        if npoint < 1:
            raise ValueError(f"npoint must be >= 1, got {npoint}")

        self.in_channels = in_channels
        self.npoint = npoint
        self.aggregation: AggregationMode = aggregation
        self.deterministic_start = deterministic_start

        self.mrg: MultiResolutionPointNetLocal | None = None
        self.msg: MultiScalePointNetLocal | None = None

        if aggregation == "multiresolution":
            if coarse_npoint is None:
                raise ValueError("coarse_npoint is required when aggregation='multiresolution'")
            if coarse_npoint < 1:
                raise ValueError(f"coarse_npoint must be >= 1, got {coarse_npoint}")
            fine_mlp_dims = [32, 64] if fine_mlp_dims is None else fine_mlp_dims
            coarse_mlp_dims = [32, 64] if coarse_mlp_dims is None else coarse_mlp_dims
            self.coarse_npoint = coarse_npoint
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
        elif aggregation == "multiscale":
            if msg_radii is None or msg_max_neighbors is None or msg_mlp_dims_per_scale is None:
                raise ValueError(
                    "msg_radii, msg_max_neighbors, and msg_mlp_dims_per_scale are required "
                    "when aggregation='multiscale'"
                )
            self.coarse_npoint = None
            self.msg = MultiScalePointNetLocal(
                in_channels=in_channels,
                radii=msg_radii,
                max_neighbors=msg_max_neighbors,
                mlp_dims_per_scale=msg_mlp_dims_per_scale,
                npoint=npoint,
                deterministic_start=deterministic_start,
            )
        else:
            raise ValueError(
                f"aggregation must be 'multiresolution' or 'multiscale', got {aggregation!r}"
            )

    @property
    def out_channels(self) -> int:
        if self.mrg is not None:
            return self.mrg.out_channels
        assert self.msg is not None
        return self.msg.out_channels

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
        if self.aggregation == "multiresolution":
            fps_idx = farthest_point_sample(
                x, self.npoint, deterministic_start=self.deterministic_start
            )
            new_xyz = index_points(x[..., :3], fps_idx)
            coarse_idx = farthest_point_sample(
                x, self.coarse_npoint, deterministic_start=self.deterministic_start
            )
            coarse_xyz_and_feats = index_points(x, coarse_idx)
            fused, _, _w = self.mrg(
                fine_xyz=x,
                coarse_xyz=coarse_xyz_and_feats,
                centroid_xyz=new_xyz,
            )
            
        elif self.aggregation == "multiscale":
            fps_idx = farthest_point_sample(
                x, self.npoint, deterministic_start=self.deterministic_start
            )
            new_xyz = index_points(x[..., :3], fps_idx)
            fused, _ = self.msg(x, centroid_xyz=new_xyz)
        
        
        
        
        return new_xyz, fused
