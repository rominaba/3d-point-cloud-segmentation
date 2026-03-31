import pytest
import torch

from src.pointnetpp.multiscale import MultiScalePointNetLocal


def test_msg_shape_with_given_centroids():
    b, n, s, c = 2, 64, 8, 6
    xyz = torch.randn(b, n, c)
    centroids = torch.randn(b, s, 3)
    msg = MultiScalePointNetLocal(
        in_channels=c,
        radii=[0.1, 0.2, 0.4],
        max_neighbors=[8, 16, 32],
        mlp_dims_per_scale=[[16, 32], [32, 64], [64, 64]],
    )
    feat, c_out = msg(xyz, centroids)
    assert feat.shape == (b, s, 32 + 64 + 64)
    assert c_out.shape == (b, s, 3)
    assert msg.out_channels == 160


def test_msg_fps_centroids_when_none():
    b, n, c = 1, 40, 3
    xyz = torch.randn(b, n, c)
    msg = MultiScalePointNetLocal(
        in_channels=c,
        radii=[0.2, 0.5],
        max_neighbors=[8, 16],
        mlp_dims_per_scale=[[16, 32], [32, 48]],
        npoint=5,
        deterministic_start=True,
    )
    feat, centroids = msg(xyz, centroid_xyz=None)
    assert feat.shape == (b, 5, 32 + 48)
    assert centroids.shape == (b, 5, 3)


def test_msg_bad_config_lengths():
    with pytest.raises(ValueError, match="same length"):
        MultiScalePointNetLocal(
            in_channels=3,
            radii=[0.1, 0.2],
            max_neighbors=[8],
            mlp_dims_per_scale=[[16], [32]],
        )


def test_msg_missing_centroids_and_npoint():
    msg = MultiScalePointNetLocal(
        in_channels=3,
        radii=[0.1],
        max_neighbors=[8],
        mlp_dims_per_scale=[[16]],
        npoint=None,
    )
    with pytest.raises(ValueError, match="npoint"):
        msg(torch.randn(1, 10, 3), centroid_xyz=None)
