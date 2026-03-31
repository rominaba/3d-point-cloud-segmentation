"""
Simple PointNet++ part segmentation model (encoder + feature propagation + head).

This implementation uses:
  - Set Abstraction layers built on MultiResolutionPointNetLocal (MRG)
  - Feature Propagation (inverse-distance interpolation)
  - Per-point MLP classifier
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .feature_propagation import FeaturePropagation
from .set_abstraction import SetAbstractionMRG


def _make_shared_mlp(in_dim: int, mlp_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    c = in_dim
    for d in mlp_dims:
        layers.append(nn.Linear(c, d))
        layers.append(nn.ReLU(inplace=True))
        c = d
    return nn.Sequential(*layers)


class PointNetPPPartSeg(nn.Module):
    """
    Forward expects point tensors shaped (B, N, C) where:
      - first 3 channels are XYZ
      - remaining channels are optional features (e.g. normals)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_part_classes: int,
        # Optional category conditioning (PointNet++ part_seg style).
        num_categories: int | None = None,
        category_embed_dim: int = 0,
        # Encoder SA/MRG settings.
        sa1_npoint: int = 256,
        sa1_coarse_npoint: int = 128,
        sa1_fine_radius: float = 0.2,
        sa1_fine_max_neighbors: int = 16,
        sa1_fine_mlp_dims: list[int] = None,  # type: ignore[assignment]
        sa1_coarse_radius: float = 0.4,
        sa1_coarse_max_neighbors: int = 32,
        sa1_coarse_mlp_dims: list[int] = None,  # type: ignore[assignment]
        sa1_fused_dim: int = 128,
        sa2_npoint: int = 64,
        sa2_coarse_npoint: int = 32,
        sa2_fine_radius: float = 0.4,
        sa2_fine_max_neighbors: int = 16,
        sa2_fine_mlp_dims: list[int] = None,  # type: ignore[assignment]
        sa2_coarse_radius: float = 0.8,
        sa2_coarse_max_neighbors: int = 32,
        sa2_coarse_mlp_dims: list[int] = None,  # type: ignore[assignment]
        sa2_fused_dim: int = 256,
        # Decoder FP settings.
        fp_k: int = 3,
        fp2_mlp_dims: list[int] = None,  # type: ignore[assignment]
        fp1_mlp_dims: list[int] = None,  # type: ignore[assignment]
        # Classifier
        head_mlp_dims: list[int] = None,  # type: ignore[assignment]
    ) -> None:
        super().__init__()
        if in_channels < 3:
            raise ValueError(f"in_channels must be >= 3 (xyz + extras), got {in_channels}")
        if num_part_classes < 1:
            raise ValueError(f"num_part_classes must be >= 1, got {num_part_classes}")

        # Default dims (kept small and configurable).
        sa1_fine_mlp_dims = [32, 64] if sa1_fine_mlp_dims is None else sa1_fine_mlp_dims
        sa1_coarse_mlp_dims = (
            [32, 64] if sa1_coarse_mlp_dims is None else sa1_coarse_mlp_dims
        )
        sa2_fine_mlp_dims = [64, 128] if sa2_fine_mlp_dims is None else sa2_fine_mlp_dims
        sa2_coarse_mlp_dims = (
            [64, 128] if sa2_coarse_mlp_dims is None else sa2_coarse_mlp_dims
        )

        fp2_mlp_dims = [128, 128] if fp2_mlp_dims is None else fp2_mlp_dims
        fp1_mlp_dims = [128, 128] if fp1_mlp_dims is None else fp1_mlp_dims
        head_mlp_dims = [128, 64] if head_mlp_dims is None else head_mlp_dims

        # Encoder.
        self.sa1 = SetAbstractionMRG(
            in_channels=in_channels,
            npoint=sa1_npoint,
            coarse_npoint=sa1_coarse_npoint,
            fine_radius=sa1_fine_radius,
            fine_max_neighbors=sa1_fine_max_neighbors,
            fine_mlp_dims=sa1_fine_mlp_dims,
            coarse_radius=sa1_coarse_radius,
            coarse_max_neighbors=sa1_coarse_max_neighbors,
            coarse_mlp_dims=sa1_coarse_mlp_dims,
            fused_dim=sa1_fused_dim,
        )
        self.sa2 = SetAbstractionMRG(
            in_channels=3 + self.sa1.out_channels,
            npoint=sa2_npoint,
            coarse_npoint=sa2_coarse_npoint,
            fine_radius=sa2_fine_radius,
            fine_max_neighbors=sa2_fine_max_neighbors,
            fine_mlp_dims=sa2_fine_mlp_dims,
            coarse_radius=sa2_coarse_radius,
            coarse_max_neighbors=sa2_coarse_max_neighbors,
            coarse_mlp_dims=sa2_coarse_mlp_dims,
            fused_dim=sa2_fused_dim,
        )

        # Decoder.
        # FP2: from SA2 (coarse) to SA1 resolution, concat with SA1 features.
        fp2_in_channels = sa2_fused_dim + sa1_fused_dim
        self.fp2 = FeaturePropagation(
            in_channels=fp2_in_channels,
            k=fp_k,
            mlp_dims=fp2_mlp_dims,
        )
        # FP1: from SA1 to original points, concat with input skip features.
        fp1_in_channels = fp2_mlp_dims[-1] + in_channels
        self.fp1 = FeaturePropagation(
            in_channels=fp1_in_channels,
            k=fp_k,
            mlp_dims=fp1_mlp_dims,
        )

        # Optional category embedding.
        self.use_category_embed = (
            num_categories is not None and category_embed_dim is not None and category_embed_dim > 0
        )
        self.num_categories = num_categories
        self.category_embed_dim = category_embed_dim if self.use_category_embed else 0
        if self.use_category_embed:
            if num_categories is None:
                raise ValueError("num_categories must be set when using category embedding")
            self.category_embed = nn.Embedding(num_categories, category_embed_dim)
        else:
            self.category_embed = None

        # Classifier MLP applied per point.
        # Input dim to head will be fp1_out_dim + optional category embedding.
        fp1_out_dim = fp1_mlp_dims[-1]
        head_in_dim = fp1_out_dim + self.category_embed_dim
        self.head = _make_shared_mlp(head_in_dim, head_mlp_dims)
        self.classifier = nn.Linear(head_mlp_dims[-1], num_part_classes)

    def forward(
        self,
        x: torch.Tensor,
        class_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, C_in) with first 3 channels XYZ.
            class_label: (B,) long category id for optional category conditioning.

        Returns:
            logits: (B, N, num_part_classes)
        """
        if x.dim() != 3:
            raise ValueError(f"x must be (B, N, C), got {tuple(x.shape)}")
        if x.size(-1) < 3:
            raise ValueError("x must have at least 3 channels (xyz)")

        xyz0 = x[..., :3]
        feats0 = x  # keep coords+extras as skip features for simplicity

        # Encoder: SA1
        xyz1, feats1 = self.sa1(x)
        point_tensor1 = torch.cat([xyz1, feats1], dim=-1)  # (B, S1, 3+C1)

        # Encoder: SA2
        xyz2, feats2 = self.sa2(point_tensor1)

        # Decoder: FP2 to xyz1
        feats1_up = self.fp2(
            xyz_coarse=xyz2,
            feats_coarse=feats2,
            xyz_fine=xyz1,
            feats_fine_skip=feats1,
        )

        # Decoder: FP1 to original xyz0
        feats0_up = self.fp1(
            xyz_coarse=xyz1,
            feats_coarse=feats1_up,
            xyz_fine=xyz0,
            feats_fine_skip=feats0,
        )

        # Optional category conditioning (zeros if class_label omitted).
        if self.use_category_embed:
            b, n, _ = feats0_up.shape
            if class_label is None:
                cat = torch.zeros(
                    b,
                    n,
                    self.category_embed_dim,
                    device=feats0_up.device,
                    dtype=feats0_up.dtype,
                )
            else:
                cat = self.category_embed(class_label).unsqueeze(1).expand(-1, n, -1)
            feats0_up = torch.cat([feats0_up, cat], dim=-1)

        h = self.head(feats0_up)
        logits = self.classifier(h)
        return logits

