"""
Simple PointNet++ classification model (encoder + global pooling + classifier head).
"""

from __future__ import annotations

from typing import Literal
import torch
import torch.nn as nn

from .set_abstraction import SetAbstraction

SaAggregation = Literal["multiresolution", "multiscale"]


def _make_mlp(in_dim: int, dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    c = in_dim
    for d in dims:
        layers.append(nn.Linear(c, d))
        layers.append(nn.ReLU(inplace=True))
        c = d
    return nn.Sequential(*layers)


class PointNetPPClassifier(nn.Module):
    """
    Point cloud classifier with PointNet++-style set abstraction encoder.

    Input:
        x: (B, N, C), C >= 3 and first 3 dims are XYZ.
    Output:
        logits: (B, num_classes)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        sa_aggregation: SaAggregation = "multiresolution",
        # Abstraction layers for MultiResolution
        sa1_npoint: int = 256,
        sa1_coarse_npoint: int = 128,
        sa1_fine_radius: float = 0.2,
        sa1_fine_max_neighbors: int = 16,
        sa1_fine_mlp_dims: list[int] = [32, 64],
        sa1_coarse_radius: float = 0.4,
        sa1_coarse_max_neighbors: int = 32,
        sa1_coarse_mlp_dims: list[int] = [32, 64],
        sa1_fused_dim: int = 128,
        sa2_npoint: int = 64,
        sa2_coarse_npoint: int = 32,
        sa2_fine_radius: float = 0.4,
        sa2_fine_max_neighbors: int = 16,
        sa2_fine_mlp_dims: list[int] = [64, 128],
        sa2_coarse_radius: float = 0.8,
        sa2_coarse_max_neighbors: int = 32,
        sa2_coarse_mlp_dims: list[int] = [64, 128],
        sa2_fused_dim: int = 256,
        # Abstraction layers for MultiScale
        sa1_msg_radii: list[float] = [0.2, 0.4],
        sa1_msg_max_neighbors: list[int] = [16, 32],
        sa1_msg_mlp_dims_per_scale: list[list[int]] = [[32, 64], [32, 64]],
        sa2_msg_radii: list[float] = [0.4, 0.8],
        sa2_msg_max_neighbors: list[int] = [16, 32],
        sa2_msg_mlp_dims_per_scale: list[list[int]] = [[64, 128], [64, 128]],
        head_mlp_dims: list[int] = [256, 128],
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.sa_aggregation = sa_aggregation

        if sa_aggregation == "multiresolution":
            self.sa1 = SetAbstraction(
                in_channels=in_channels,
                npoint=sa1_npoint,
                aggregation="multiresolution",
                coarse_npoint=sa1_coarse_npoint,
                fine_radius=sa1_fine_radius,
                fine_max_neighbors=sa1_fine_max_neighbors,
                fine_mlp_dims=sa1_fine_mlp_dims,
                coarse_radius=sa1_coarse_radius,
                coarse_max_neighbors=sa1_coarse_max_neighbors,
                coarse_mlp_dims=sa1_coarse_mlp_dims,
                fused_dim=sa1_fused_dim,
            )
        else:
            
            self.sa1 = SetAbstraction(
                in_channels=in_channels,
                npoint=sa1_npoint,
                aggregation="multiscale",
                msg_radii=sa1_msg_radii,
                msg_max_neighbors=sa1_msg_max_neighbors,
                msg_mlp_dims_per_scale=sa1_msg_mlp_dims_per_scale,
            )

        sa2_in = 3 + self.sa1.out_channels
        if sa_aggregation == "multiresolution":
            self.sa2 = SetAbstraction(
                in_channels=sa2_in,
                npoint=sa2_npoint,
                aggregation="multiresolution",
                coarse_npoint=sa2_coarse_npoint,
                fine_radius=sa2_fine_radius,
                fine_max_neighbors=sa2_fine_max_neighbors,
                fine_mlp_dims=sa2_fine_mlp_dims,
                coarse_radius=sa2_coarse_radius,
                coarse_max_neighbors=sa2_coarse_max_neighbors,
                coarse_mlp_dims=sa2_coarse_mlp_dims,
                fused_dim=sa2_fused_dim,
            )
        else:
            r2 = sa2_msg_radii if sa2_msg_radii is not None else [0.4, 0.8]
            k2 = sa2_msg_max_neighbors if sa2_msg_max_neighbors is not None else [16, 32]
            m2 = (
                sa2_msg_mlp_dims_per_scale
                if sa2_msg_mlp_dims_per_scale is not None
                else [[64, 128], [64, 128]]
            )
            self.sa2 = SetAbstraction(
                in_channels=sa2_in,
                npoint=sa2_npoint,
                aggregation="multiscale",
                msg_radii=r2,
                msg_max_neighbors=k2,
                msg_mlp_dims_per_scale=m2,
            )
        self.head = _make_mlp(self.sa2.out_channels, head_mlp_dims)
        self.classifier = nn.Linear(head_mlp_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xyz1, feat1 = self.sa1(x)
        x1 = torch.cat([xyz1, feat1], dim=-1)

        _xyz2, feat2 = self.sa2(x1)
        g = torch.max(feat2, dim=1).values

        h = self.head(g)
        logits = self.classifier(h)
        return logits

