"""
PointNet++-style Multi-Resolution Grouping (MRG) using PointNetLocal blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .grouping import ball_query_group_relative
from .pointnet import PointNetLocal
from .sampling import farthest_point_sample, index_points


class MultiResolutionPointNetLocal(nn.Module):
    """
    Fuse fine- and coarse-resolution local features around shared centroids.

    The module computes one local feature from a fine point set and one from a
    coarse point set, then fuses them with simple density-adaptive weights.
    """

    def __init__(
        self,
        in_channels: int,
        fine_radius: float,
        fine_max_neighbors: int,
        fine_mlp_dims: list[int],
        coarse_radius: float,
        coarse_max_neighbors: int,
        coarse_mlp_dims: list[int],
        # Number of centroids
        npoint: int | None = None,
        deterministic_start: bool = False,
        fused_dim: int | None = None,
    ) -> None:
        super().__init__()
        if npoint is not None and npoint < 1:
            raise ValueError(f"npoint must be >= 1 when provided, got {npoint}")

        self.in_channels = in_channels
        self.fine_radius = fine_radius
        self.fine_max_neighbors = fine_max_neighbors
        self.coarse_radius = coarse_radius
        self.coarse_max_neighbors = coarse_max_neighbors
        self.npoint = npoint
        self.deterministic_start = deterministic_start

        self.fine_net = PointNetLocal(in_channels, fine_mlp_dims)
        self.coarse_net = PointNetLocal(in_channels, coarse_mlp_dims)

        fd = fused_dim if fused_dim is not None else max(
            self.fine_net.out_channels, self.coarse_net.out_channels
        )
        if fd < 1:
            raise ValueError(f"fused_dim must be >= 1, got {fd}")
        self._out_channels = fd

        self.fine_proj = nn.Linear(self.fine_net.out_channels, fd)
        self.coarse_proj = nn.Linear(self.coarse_net.out_channels, fd)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(
        self,
        fine_xyz: torch.Tensor,
        coarse_xyz: torch.Tensor,
        centroid_xyz: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            fine_xyz: Higher resolution point cloud (B, Nf, C), C >= 3. 
                Where B is the batch size, Nf is the number of points in the fine point cloud, 
                and C is the number of channels/features per point.
            coarse_xyz: Lower resolution point cloud (B, Nc, C), C >= 3.
                Where B is the batch size, Nc is the number of points in the coarse point cloud, 
                and C is the number of channels/features per point.
            centroid_xyz: Optional (B, S, 3). If None, centroids are FPS on fine_xyz.

        Returns:
            fused: (B, S, out_channels), fused MRG features.
            centroid_xyz: (B, S, 3), provided or sampled centroids.
            weights: (B, S, 2), [w_fine, w_coarse] used for fusion.
        """
        if fine_xyz.dim() != 3 or coarse_xyz.dim() != 3:
            raise ValueError("fine_xyz and coarse_xyz must be (B, N, C)")
        if fine_xyz.size(-1) < 3 or coarse_xyz.size(-1) < 3:
            raise ValueError("fine_xyz and coarse_xyz must have C >= 3")
        if fine_xyz.size(0) != coarse_xyz.size(0):
            raise ValueError(
                f"batch mismatch: fine B={fine_xyz.size(0)}, coarse B={coarse_xyz.size(0)}"
            )

        b = fine_xyz.size(0)
        if centroid_xyz is None:
            if self.npoint is None:
                raise ValueError("centroid_xyz is None, but npoint was not set")
            fps_idx = farthest_point_sample(
                fine_xyz, self.npoint, deterministic_start=self.deterministic_start
            )
            centroid_xyz = index_points(fine_xyz[..., :3], fps_idx)
        else:
            if centroid_xyz.dim() != 3 or centroid_xyz.size(-1) != 3:
                raise ValueError(
                    f"centroid_xyz must be (B, S, 3), got {tuple(centroid_xyz.shape)}"
                )
            if centroid_xyz.size(0) != b:
                raise ValueError(
                    f"batch mismatch: fine B={b}, centroid_xyz B={centroid_xyz.size(0)}"
                )

        fine_rel, _, fine_mask = ball_query_group_relative(
            fine_xyz, centroid_xyz, radius=self.fine_radius, max_neighbors=self.fine_max_neighbors
        )
        coarse_rel, _, coarse_mask = ball_query_group_relative(
            coarse_xyz,
            centroid_xyz,
            radius=self.coarse_radius,
            max_neighbors=self.coarse_max_neighbors,
        )

        fine_feat = self.fine_proj(self.fine_net(fine_rel, mask=fine_mask))
        coarse_feat = self.coarse_proj(self.coarse_net(coarse_rel, mask=coarse_mask))

        # Density proxy in [0, 1]: fraction of valid neighbors per centroid.
        fine_den = fine_mask.float().mean(dim=-1, keepdim=True)
        coarse_den = coarse_mask.float().mean(dim=-1, keepdim=True)

        # If fine neighborhoods are sparse, rely more on coarse features.
        eps = 1e-6
        inv_fine = 1.0 - fine_den
        inv_coarse = 1.0 - coarse_den
        w_coarse = inv_fine / (inv_fine + inv_coarse + eps)
        w_fine = 1.0 - w_coarse

        fused = w_fine * fine_feat + w_coarse * coarse_feat
        weights = torch.cat([w_fine, w_coarse], dim=-1)
        return fused, centroid_xyz, weights
