import pytest
import torch

from src.pointnetpp.multiresolution import MultiResolutionPointNetLocal


def test_mrg_shape_with_given_centroids():
    b, nf, nc, s, c = 2, 64, 16, 10, 6
    fine_xyz = torch.randn(b, nf, c)
    coarse_xyz = torch.randn(b, nc, c)
    centroids = torch.randn(b, s, 3)

    mrg = MultiResolutionPointNetLocal(
        in_channels=c,
        fine_radius=0.1,
        fine_max_neighbors=16,
        fine_mlp_dims=[32, 64],
        coarse_radius=0.3,
        coarse_max_neighbors=8,
        coarse_mlp_dims=[16, 32],
        fused_dim=48,
    )
    fused, c_out, w = mrg(fine_xyz, coarse_xyz, centroids)
    assert fused.shape == (b, s, 48)
    assert c_out.shape == (b, s, 3)
    assert w.shape == (b, s, 2)
    assert mrg.out_channels == 48


def test_mrg_fps_centroids_when_none():
    fine_xyz = torch.randn(1, 40, 3)
    coarse_xyz = torch.randn(1, 12, 3)
    mrg = MultiResolutionPointNetLocal(
        in_channels=3,
        fine_radius=0.2,
        fine_max_neighbors=12,
        fine_mlp_dims=[16, 32],
        coarse_radius=0.5,
        coarse_max_neighbors=6,
        coarse_mlp_dims=[16, 24],
        npoint=5,
        deterministic_start=True,
    )
    fused, centroids, w = mrg(fine_xyz, coarse_xyz, centroid_xyz=None)
    assert fused.shape == (1, 5, mrg.out_channels)
    assert centroids.shape == (1, 5, 3)
    assert w.shape == (1, 5, 2)


def test_mrg_weights_sum_to_one():
    fine_xyz = torch.randn(1, 32, 3)
    coarse_xyz = torch.randn(1, 16, 3)
    centroids = torch.randn(1, 6, 3)
    mrg = MultiResolutionPointNetLocal(
        in_channels=3,
        fine_radius=0.2,
        fine_max_neighbors=8,
        fine_mlp_dims=[16],
        coarse_radius=0.4,
        coarse_max_neighbors=6,
        coarse_mlp_dims=[16],
    )
    _, _, w = mrg(fine_xyz, coarse_xyz, centroids)
    ones = torch.ones_like(w[..., 0])
    assert torch.allclose(w[..., 0] + w[..., 1], ones, atol=1e-5)


def test_mrg_missing_centroids_and_npoint():
    mrg = MultiResolutionPointNetLocal(
        in_channels=3,
        fine_radius=0.1,
        fine_max_neighbors=8,
        fine_mlp_dims=[16],
        coarse_radius=0.2,
        coarse_max_neighbors=8,
        coarse_mlp_dims=[16],
        npoint=None,
    )
    with pytest.raises(ValueError, match="npoint"):
        mrg(torch.randn(1, 10, 3), torch.randn(1, 5, 3), centroid_xyz=None)
