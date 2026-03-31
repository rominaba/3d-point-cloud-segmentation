"""
PointNet++ Multi-Scale Grouping (MSG) with PointNetLocal feature extractors.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .grouping import ball_query_group_relative
from .pointnet import PointNetLocal
from .sampling import farthest_point_sample, index_points


class MultiScalePointNetLocal(nn.Module):
    """
    Multi-scale local feature extraction around centroids.

    For each centroid and each scale j:
      1) group neighbors inside radius[j]
      2) run PointNetLocal_j on grouped neighbors
      3) concatenate scale features along channels
    """

    def __init__(
        self,
        in_channels: int,
        radii: list[float],
        max_neighbors: list[int],
        mlp_dims_per_scale: list[list[int]],
        *,
        npoint: int | None = None,
        deterministic_start: bool = False,
    ) -> None:
        super().__init__()
        if len(radii) == 0:
            raise ValueError("radii must be non-empty")
        if not (len(radii) == len(max_neighbors) == len(mlp_dims_per_scale)):
            raise ValueError(
                "radii, max_neighbors, mlp_dims_per_scale must have the same length"
            )
        if npoint is not None and npoint < 1:
            raise ValueError(f"npoint must be >= 1 when provided, got {npoint}")

        self.in_channels = in_channels
        self.radii = list(radii)
        self.max_neighbors = list(max_neighbors)
        self.npoint = npoint
        self.deterministic_start = deterministic_start

        self.local_nets = nn.ModuleList(
            [PointNetLocal(in_channels, dims) for dims in mlp_dims_per_scale]
        )

    @property
    def out_channels(self) -> int:
        return sum(m.out_channels for m in self.local_nets)

    def forward(
        self, xyz: torch.Tensor, centroid_xyz: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, C), C >= 3. First 3 dims are XYZ.
            centroid_xyz: Optional (B, S, 3). If None, centroids are FPS sampled.

        Returns:
            feat: (B, S, sum_j C_j), concatenated MSG features.
            centroid_xyz: (B, S, 3), provided or FPS-sampled centroids.
        """
        if xyz.dim() != 3:
            raise ValueError(f"xyz must be (B, N, C), got {tuple(xyz.shape)}")
        if xyz.size(-1) < 3:
            raise ValueError(f"xyz needs C >= 3, got {xyz.size(-1)}")

        b, _, _ = xyz.shape

        if centroid_xyz is None:
            if self.npoint is None:
                raise ValueError("centroid_xyz is None, but npoint was not set")
            fps_idx = farthest_point_sample(
                xyz, self.npoint, deterministic_start=self.deterministic_start
            )
            centroid_xyz = index_points(xyz[..., :3], fps_idx)
        else:
            if centroid_xyz.dim() != 3 or centroid_xyz.size(-1) != 3:
                raise ValueError(
                    f"centroid_xyz must be (B, S, 3), got {tuple(centroid_xyz.shape)}"
                )
            if centroid_xyz.size(0) != b:
                raise ValueError(
                    f"batch mismatch: xyz B={b}, centroid_xyz B={centroid_xyz.size(0)}"
                )

        scale_feats: list[torch.Tensor] = []
        for radius, k, net in zip(self.radii, self.max_neighbors, self.local_nets):
            rel, _, mask = ball_query_group_relative(
                xyz, centroid_xyz, radius=radius, max_neighbors=k
            )
            scale_feats.append(net(rel, mask=mask))

        feat = torch.cat(scale_feats, dim=-1)
        return feat, centroid_xyz
