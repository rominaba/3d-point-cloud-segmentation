"""
Ball query grouping for PointNet++: neighbors within a radius, in centroid-relative coordinates.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .sampling import _xyz_for_distance, index_points


def ball_query_group_relative(
    xyz: torch.Tensor,
    centroid_xyz: torch.Tensor,
    radius: float,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For each centroid, take up to max_neighbors closest points whose distance to
    that centroid is at most radius. Return offsets x_i - x'_j for those points.

    Args:
        xyz: Full point set, shape (B, N, C) with C >= 3. Distances use the
            first three channels only.
        centroid_xyz: Centroid positions, shape (B, S, 3).
        radius: Ball radius (Euclidean, same units as coordinates).
        max_neighbors: Upper bound K on how many points to keep per centroid
            (fixed tensor shapes).

    Returns:
        rel: (B, S, K, C) — gathered point features with the first three channels
            replaced by x_i - x'_j; remaining channels are unchanged if C > 3.
        idx: (B, S, K) long — index into N for each slot.
        mask: (B, S, K) bool — True if that slot is a valid in-radius neighbor.
    """
    if xyz.dim() != 3 or centroid_xyz.dim() != 3:
        raise ValueError("xyz must be (B, N, C) and centroid_xyz (B, S, 3)")
    # Extract shape information from the input point cloud:
    # B = batch size, N = number of points, C = number of channels/features
    b, n, c = xyz.shape
    # Extract shape information from the centroid tensor:
    # B2 = batch size, S = number of centroids, C3 should always be 3 (x,y,z)
    b2, s, c3 = centroid_xyz.shape
    if b != b2:
        raise ValueError(f"batch mismatch: xyz B={b}, centroid_xyz B={b2}")
    if c3 != 3:
        raise ValueError(f"centroid_xyz last dim must be 3, got {c3}")
    if c < 3:
        raise ValueError(f"xyz needs C >= 3, got {c}")
    if max_neighbors < 1:
        raise ValueError(f"max_neighbors must be >= 1, got {max_neighbors}")
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")

    k_take = min(max_neighbors, n)
    xyz_d = _xyz_for_distance(xyz)
    # Compute squared Euclidean distances from each centroid to every point.
    # Shapes:
    # xyz_d.unsqueeze(1)        -> (B, 1, N, 3)
    # centroid_xyz.unsqueeze(2) -> (B, S, 1, 3)
    # subtraction broadcasts    -> (B, S, N, 3)
    # sum over last dim         -> (B, S, N)
    #
    # Result: dist_sq[b, j, i] = squared distance from point i to centroid j in batch b
    dist_sq = torch.sum(
        (xyz_d.unsqueeze(1) - centroid_xyz.unsqueeze(2)) ** 2, dim=-1
    )
    r2 = radius * radius
    in_ball = dist_sq <= r2
    # Replace outsiders with a valid value in case the real eligible points are less than max_neighbors
    big = torch.finfo(dist_sq.dtype).max / 4
    dist_sort = dist_sq.masked_fill(~in_ball, big)
    # Select the k_take smallest distances for each centroid.
    # Since largest=False, this gives nearest points first.
    # If there are fewer than k_take valid in-ball points, topk will still fill
    # remaining slots using the smallest "big" values (i.e. out-of-ball points).
    _, idx = torch.topk(dist_sort, k_take, dim=-1, largest=False)
    # Recover which of the selected indices are actually valid in-ball neighbors
    mask = torch.gather(in_ball, 2, idx)
    # If max_neighbors > N, pad idx and mask so output shape is always fixed
    if k_take < max_neighbors:
        pad = max_neighbors - k_take
        z = idx.new_zeros(b, s, pad)
        # Append padded indices to the end
        idx = torch.cat([idx, z], dim=-1)
        # Append False mask entries for padded slots
        mask = torch.cat(
            [mask, torch.zeros(b, s, pad, dtype=torch.bool, device=mask.device)],
            dim=-1,
        )
    # Flatten the indices and gather the points
    idx_gather = idx.reshape(b, s * max_neighbors)
    # Gather the points from the original point cloud using the flattened indices
    grouped = index_points(xyz, idx_gather).view(b, s, max_neighbors, c)
    # Expand the centroid coordinates to match the shape of the grouped points
    # and subtract them from the grouped points to get the relative offsets
    c_exp = centroid_xyz.unsqueeze(2)
    # Clone the grouped points to avoid modifying the original
    rel = grouped.clone()
    # Replace the first three channels with the relative offsets
    rel[..., :3] = grouped[..., :3] - c_exp

    return rel, idx, mask


class BallQueryGroup(nn.Module):
    """Module wrapper for ball_query_group_relative()."""

    def __init__(self, radius: float, max_neighbors: int) -> None:
        super().__init__()
        self.radius = radius
        self.max_neighbors = max_neighbors

    def forward(
        self, xyz: torch.Tensor, centroid_xyz: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return ball_query_group_relative(
            xyz, centroid_xyz, self.radius, self.max_neighbors
        )
