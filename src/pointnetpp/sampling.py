"""
PointNet++-style farthest point sampling (FPS).

FPS selects a subset of input points as centroids by iteratively choosing the
point farthest from the set already chosen. Training uses the selected indices as a hard discrete choice.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _xyz_for_distance(xyz: torch.Tensor) -> torch.Tensor:
    """Use the first three channels as coordinates for distance (paper: XYZ)."""
    if xyz.size(-1) > 3:
        return xyz[..., :3]
    return xyz


def farthest_point_sample(
    xyz: torch.Tensor,
    npoint: int,
    *,
    deterministic_start: bool = False,
) -> torch.Tensor:
    """
    Batched farthest point sampling on the last spatial dimension.

    Args:
        xyz: Point coordinates, shape (B, N, C), where B is the number of point clouds in a minibatch,
            N is the number of points in each point cloud, C is the number of channels/features per points and it's >= 3. 
            If C > 3, only the first three channels are used for distances; indices still refer to full rows of xyz.
        npoint: Number of centroids N'. Must satisfy 0 <= npoint <= N.
        deterministic_start: If True, the first centroid index is 0 for every
            batch item. If False, the first index is drawn uniformly from
            [0, N) per batch item (same device as xyz; not seeded—use
            torch.manual_seed / generators before calling for reproducibility).

    Returns:
        Long tensor of shape (B, npoint) with indices into N for each batch.

    Raises:
        ValueError: If npoint is negative or npoint > N.
    """
    if npoint < 0:
        raise ValueError(f"npoint must be non-negative, got {npoint}")
    if xyz.dim() != 3:
        raise ValueError(f"xyz must be (B, N, C), got shape {(xyz.shape)}")
    b, n, c = xyz.shape
    if c < 3:
        raise ValueError(f"xyz must have at least 3 channels for FPS, got C={c}")
    if npoint > n:
        raise ValueError(f"npoint ({npoint}) must be <= N ({n})")

    if npoint == 0:
        return torch.empty(b, 0, dtype=torch.long, device=xyz.device)

    device = xyz.device
    # Use the same dtype for next calculations to avoid dtype mismatch
    dtype = _xyz_for_distance(xyz).dtype
    # Extract coordinate channels
    xyz_d = _xyz_for_distance(xyz)
    # Initialize the tensor that will store the indices of the centroids for each batch
    centroids = torch.zeros(b, npoint, dtype=torch.long, device=device)
    # Initialize the tensor that will store the distances to the centroids for each point for each batch
    distance = torch.full((b, n), float("inf"), dtype=dtype, device=device)
    # Choose the first point to start FPS from
    if deterministic_start:
        farthest = torch.zeros(b, dtype=torch.long, device=device)
    else:
        farthest = torch.randint(0, n, (b,), dtype=torch.long, device=device)
    # Create an index for each batch
    batch_arange = torch.arange(b, dtype=torch.long, device=device)
    # Example:
    #   batch_arange = [0, 1, 2]
    #   farthest     = [5, 8, 3]
    #
    # Then:
    #   xyz_d[batch_arange, farthest]
    #
    # gives:
    #   [xyz_d[0, 5], xyz_d[1, 8], xyz_d[2, 3]]
    for i in range(npoint):
        # For all batches select the ith index of farthest as the ith centroid
        centroids[:, i] = farthest
        # Get the coordinates of the ith centroid for each batch
        centroid = xyz_d[batch_arange, farthest].unsqueeze(1)
        # Compute the Exuclidean distance every point and the selected centroid in each batch
        dist = torch.sum((xyz_d - centroid) ** 2, dim=-1)
        # Determine how far each point is from the closest centroid selected for far in each batch
        distance = torch.minimum(distance, dist)
        # Choose the point whose nearest selected centroid is farthest away.
        # This point is the least well covered by the current set of centroids so far.
        # So, it is the best next point 
        farthest = torch.argmax(distance, dim=-1)

    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Returns the certoids for each batch given the indices.

    Args:
        points: (B, N, C)
        idx: (B, S) long indices into N

    Returns:
        Tensor of shape (B, S, C)
    """
    if points.dim() != 3:
        raise ValueError(f"points must be (B, N, C), got {points.shape}")
    if idx.dim() != 2:
        raise ValueError(f"idx must be (B, S), got {idx.shape}")
    b, n, _ = points.shape
    if idx.shape[0] != b:
        raise ValueError(
            f"batch size mismatch: points B={b}, idx B={idx.shape[0]}"
        )
    device = points.device
    batch = torch.arange(b, device=device, dtype=idx.dtype).unsqueeze(1).expand_as(
        idx
    )
    return points[batch, idx]


class FarthestPointSampler(nn.Module):
    """nn.Module wrapper around farthest_point_sample()."""

    def __init__(
        self,
        npoint: int,
        *,
        deterministic_start: bool = False,
    ) -> None:
        super().__init__()
        self.npoint = npoint
        self.deterministic_start = deterministic_start

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return farthest_point_sample(
            xyz,
            self.npoint,
            deterministic_start=self.deterministic_start,
        )
