"""
T-Net is a neural network within PointNet that learns a transformation matrix to transform the input point cloud to a canonical space,
in order to make PointNet robust to geometric variations.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_CONV_DIMS = (64, 128, 1024)
_DEFAULT_FC_DIMS = (512, 256)


def _normalize_tnet_widths(
    conv_dims: Sequence[int] | None,
    fc_dims: Sequence[int] | None,
) -> tuple[tuple[int, int, int], tuple[int, int]]:
    c = tuple(_DEFAULT_CONV_DIMS if conv_dims is None else conv_dims)
    f = tuple(_DEFAULT_FC_DIMS if fc_dims is None else fc_dims)
    if len(c) != 3:
        raise ValueError(f"conv_dims must have length 3, got {len(c)}")
    if len(f) != 2:
        raise ValueError(f"fc_dims must have length 2, got {len(f)}")
    for d in c + f:
        if d < 1:
            raise ValueError(f"all conv_dims and fc_dims must be >= 1, got {d}")
    return (c[0], c[1], c[2]), (f[0], f[1])


class InputTNet(nn.Module):
    """
    Predicts a square transform from global point features.

    Args:
        in_channels: Per-point input dimension (e.g. 3 for xyz).
        matrix_dim: Side length of the output matrix (e.g. 3 for 3×3).
        conv_dims: Three output channel widths for the 1×1 conv stack
            (default (64, 128, 1024), as in PointNet).
        fc_dims: Two hidden sizes for the MLP after max-pooling
            (default (512, 256)).
    """

    def __init__(
        self,
        in_channels: int = 3,
        matrix_dim: int = 3,
        conv_dims: Sequence[int] | None = None,
        fc_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if matrix_dim < 1:
            raise ValueError(f"matrix_dim must be >= 1, got {matrix_dim}")

        (c1, c2, c3), (f1, f2) = _normalize_tnet_widths(conv_dims, fc_dims)

        self.in_channels = in_channels
        self.matrix_dim = matrix_dim
        out_flat = matrix_dim * matrix_dim

        self.conv1 = nn.Conv1d(in_channels, c1, 1)
        self.conv2 = nn.Conv1d(c1, c2, 1)
        self.conv3 = nn.Conv1d(c2, c3, 1)

        self.bn1 = nn.BatchNorm1d(c1)
        self.bn2 = nn.BatchNorm1d(c2)
        self.bn3 = nn.BatchNorm1d(c3)

        self.fc1 = nn.Linear(c3, f1)
        self.fc2 = nn.Linear(f1, f2)
        self.fc3 = nn.Linear(f2, out_flat)

        self.bn4 = nn.BatchNorm1d(f1)
        self.bn5 = nn.BatchNorm1d(f2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, N]

        Returns:
            Transform matrix [B, matrix_dim, matrix_dim].
        """
        if x.dim() != 3:
            raise ValueError(f"x must be (B, C_in, N), got shape {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"expected C_in={self.in_channels}, got {x.size(1)}"
            )

        batch_size = x.size(0)
        d = self.matrix_dim

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, dim=2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.eye(d, device=x.device, dtype=x.dtype).view(1, d * d)
        identity = identity.expand(batch_size, -1)
        x = x + identity

        return x.view(batch_size, d, d)


class FeatureTNet(nn.Module):
    """
    Predicts a square transform on per-point feature channels.

    Args:
        in_channels: Feature dimension per point (first conv input).
        matrix_dim: Side length of the output matrix. Defaults to in_channels
            (standard PointNet feature transform).
        conv_dims: Three 1×1 conv output widths (default (64, 128, 1024)).
        fc_dims: Two FC hidden sizes before the regression head (default (512, 256)).
    """

    def __init__(
        self,
        in_channels: int = 64,
        matrix_dim: int | None = None,
        conv_dims: Sequence[int] | None = None,
        fc_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        md = matrix_dim if matrix_dim is not None else in_channels
        if md < 1:
            raise ValueError(f"matrix_dim must be >= 1, got {md}")

        (c1, c2, c3), (f1, f2) = _normalize_tnet_widths(conv_dims, fc_dims)

        self.in_channels = in_channels
        self.matrix_dim = md
        out_flat = md * md

        self.conv1 = nn.Conv1d(in_channels, c1, 1)
        self.conv2 = nn.Conv1d(c1, c2, 1)
        self.conv3 = nn.Conv1d(c2, c3, 1)

        self.bn1 = nn.BatchNorm1d(c1)
        self.bn2 = nn.BatchNorm1d(c2)
        self.bn3 = nn.BatchNorm1d(c3)

        self.fc1 = nn.Linear(c3, f1)
        self.fc2 = nn.Linear(f1, f2)
        self.fc3 = nn.Linear(f2, out_flat)

        self.bn4 = nn.BatchNorm1d(f1)
        self.bn5 = nn.BatchNorm1d(f2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, N]

        Returns:
            [B, matrix_dim, matrix_dim]
        """
        if x.dim() != 3:
            raise ValueError(f"x must be (B, C_in, N), got shape {tuple(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"expected C_in={self.in_channels}, got {x.size(1)}"
            )

        batch_size = x.size(0)
        d = self.matrix_dim

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, dim=2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.eye(d, device=x.device, dtype=x.dtype).view(1, d * d)
        identity = identity.expand(batch_size, -1)
        x = x + identity

        return x.view(batch_size, d, d)
