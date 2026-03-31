import pytest
import torch

from src.pointnetpp.grouping import ball_query_group_relative
from src.pointnetpp.sampling import farthest_point_sample, index_points


def test_ball_query_shapes():
    b, n, s, k = 2, 64, 8, 16
    xyz = torch.randn(b, n, 3)
    c = torch.randn(b, s, 3)
    rel, idx, mask = ball_query_group_relative(xyz, c, radius=0.5, max_neighbors=k)
    assert rel.shape == (b, s, k, 3)
    assert idx.shape == (b, s, k)
    assert mask.shape == (b, s, k)


def test_ball_query_relative_offsets():
    """Single centroid at origin; neighbors should be absolute coords when centroid is 0."""
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    c = torch.zeros(1, 1, 3)
    rel, idx, mask = ball_query_group_relative(
        xyz, c, radius=1.0, max_neighbors=2
    )
    assert mask[0, 0].sum() >= 1
    for t in range(2):
        if not mask[0, 0, t]:
            continue
        i = int(idx[0, 0, t])
        expected = xyz[0, i] - c[0, 0]
        assert torch.allclose(rel[0, 0, t], expected)


def test_ball_query_in_radius_when_masked():
    xyz = torch.randn(2, 32, 3)
    c = xyz[:, :4, :].clone()
    r = 0.2
    rel, idx, mask = ball_query_group_relative(xyz, c, radius=r, max_neighbors=8)
    xyz_d = xyz[..., :3]
    for b in range(2):
        for s in range(4):
            for t in range(8):
                if not mask[b, s, t]:
                    continue
                j = int(idx[b, s, t])
                d = torch.norm(xyz_d[b, j] - c[b, s])
                assert d <= r + 1e-5


def test_ball_query_with_fps_indices():
    xyz = torch.randn(1, 40, 3)
    fps_idx = farthest_point_sample(xyz, 5, deterministic_start=True)
    c = index_points(xyz, fps_idx)
    rel, idx, mask = ball_query_group_relative(xyz, c, radius=0.3, max_neighbors=10)
    assert rel.shape[1] == 5
