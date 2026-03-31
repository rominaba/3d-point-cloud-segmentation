import pytest
import torch

from src.pointnetpp.pointnet import PointNet, PointNetLocal


def test_pointnet_global_shape_and_max():
    b, n = 2, 32
    m = PointNet(mlp1_dims=[64, 64], mlp2_dims=[128, 1024])
    x = torch.randn(b, n, 3)
    g = m(x)
    assert g.shape == (b, 1024)
    assert m.global_dim == 1024


def test_pointnet_global_no_transforms():
    m = PointNet(
        mlp1_dims=[32],
        mlp2_dims=[64],
        use_input_transform=False,
        use_feature_transform=False,
    )
    x = torch.randn(1, 8, 3)
    g = m(x)
    assert g.shape == (1, 64)


def test_pointnet_global_mask():
    m = PointNet(
        mlp1_dims=[16],
        mlp2_dims=[32],
        use_input_transform=False,
        use_feature_transform=False,
    )
    x = torch.randn(1, 5, 3)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    mask[0, 0] = True
    g = m(x, mask=mask)
    x_only = x[:, 0:1, :]
    g_only = m(x_only)
    assert torch.allclose(g, g_only, atol=1e-5)


def test_pointnet_global_perm_invariant():
    torch.manual_seed(0)
    m = PointNet(
        mlp1_dims=[32, 32],
        mlp2_dims=[64, 64],
        use_input_transform=False,
        use_feature_transform=False,
    )
    x = torch.randn(1, 16, 3)
    perm = torch.randperm(16)
    g1 = m(x)
    g2 = m(x[:, perm, :])
    assert torch.allclose(g1, g2, atol=1e-5)


def test_pointnet_global_bad_shape():
    m = PointNet(mlp1_dims=[8], mlp2_dims=[16], use_input_transform=False, use_feature_transform=False)
    with pytest.raises(ValueError, match="x must be"):
        m(torch.randn(2, 3))
    with pytest.raises(ValueError, match="last dim 3"):
        m(torch.randn(1, 4, 5))


def test_pointnet_local_shape_no_gamma():
    b, s, k, c_in = 2, 4, 16, 6
    m = PointNetLocal(c_in, [32, 64])
    x = torch.randn(b, s, k, c_in)
    y = m(x)
    assert y.shape == (b, s, 64)
    assert m.out_channels == 64


def test_pointnet_local_shape_with_gamma():
    b, s, k, c_in = 1, 3, 8, 3
    m = PointNetLocal(c_in, [16, 32], post_mlp_dims=[48, 10])
    x = torch.randn(b, s, k, c_in)
    y = m(x)
    assert y.shape == (b, s, 10)
    assert m.out_channels == 10


def test_pointnet_local_mask_ignores_invalid():
    b, s, k, c_in = 1, 1, 4, 3
    m = PointNetLocal(c_in, [16])
    x = torch.randn(b, s, k, c_in)
    mask = torch.zeros(b, s, k, dtype=torch.bool)
    mask[0, 0, 0] = True
    y = m(x, mask=mask)
    x_flat = x[0:1, 0:1, 0:1, :].reshape(1, 1, c_in)
    h_out = m.h(x_flat)
    expected = h_out.max(dim=1).values
    assert torch.allclose(y[0, 0], expected[0], atol=1e-5)


def test_pointnet_local_max_equivariance_random_subset():
    torch.manual_seed(0)
    b, s, k, c_in = 1, 2, 12, 4
    m = PointNetLocal(c_in, [24, 24])
    x = torch.randn(b, s, k, c_in)
    perm = torch.randperm(k)
    x_perm = x[:, :, perm, :]
    y1 = m(x)
    y2 = m(x_perm)
    assert torch.allclose(y1, y2, atol=1e-5)


def test_pointnet_local_bad_dims():
    m = PointNetLocal(3, [8])
    with pytest.raises(ValueError, match="x must be"):
        m(torch.randn(2, 3))
    with pytest.raises(ValueError, match="expected C_in"):
        m(torch.randn(1, 1, 4, 5))


def test_pointnet_local_empty_mlp_raises():
    with pytest.raises(ValueError, match="mlp_dims"):
        PointNetLocal(3, [])
