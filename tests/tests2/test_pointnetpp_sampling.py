import pytest
import torch

from src.pointnetpp.sampling import (
    FarthestPointSampler,
    farthest_point_sample,
    index_points,
)


def test_farthest_point_sample_shapes():
    b, n, npoint = 2, 32, 8
    xyz = torch.randn(b, n, 3)
    idx = farthest_point_sample(xyz, npoint, deterministic_start=True)
    assert idx.shape == (b, npoint)
    assert idx.dtype == torch.long
    assert (idx >= 0).all() and (idx < n).all()


def test_farthest_point_sample_npoint_zero():
    xyz = torch.randn(1, 10, 3)
    idx = farthest_point_sample(xyz, 0, deterministic_start=True)
    assert idx.shape == (1, 0)


def test_farthest_point_sample_npoint_exceeds_n():
    xyz = torch.randn(1, 5, 3)
    with pytest.raises(ValueError, match="npoint"):
        farthest_point_sample(xyz, 6, deterministic_start=True)


def test_farthest_point_sample_channels_lt_3():
    xyz = torch.randn(1, 5, 2)
    with pytest.raises(ValueError, match="at least 3"):
        farthest_point_sample(xyz, 2, deterministic_start=True)


def test_index_points_gather():
    b, n, c, s = 2, 16, 6, 4
    points = torch.randn(b, n, c)
    idx = torch.randint(0, n, (b, s))
    out = index_points(points, idx)
    assert out.shape == (b, s, c)
    for bi in range(b):
        for j in range(s):
            assert torch.equal(out[bi, j], points[bi, idx[bi, j]])


def test_index_points_batch_mismatch():
    p = torch.randn(2, 4, 3)
    idx = torch.zeros(3, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="batch size"):
        index_points(p, idx)


def test_deterministic_colinear_fps_order():
    """Known order on a line: start 0, then farthest is N-1, then interior."""
    n = 5
    xyz = torch.zeros(1, n, 3)
    xyz[0, :, 0] = torch.arange(n, dtype=torch.float32)
    idx = farthest_point_sample(xyz, n, deterministic_start=True)
    # First 0, then 4, then argmax among remaining mins — expect [0,4,1,3,2] or similar
    assert idx[0, 0].item() == 0
    assert idx[0, 1].item() == n - 1


def test_farthest_point_sampler_module():
    m = FarthestPointSampler(4, deterministic_start=True)
    xyz = torch.randn(2, 20, 3)
    idx = m(xyz)
    assert idx.shape == (2, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_farthest_point_sample_cuda():
    xyz = torch.randn(2, 64, 3, device="cuda")
    idx = farthest_point_sample(xyz, 16, deterministic_start=True)
    assert idx.device.type == "cuda"
    gathered = index_points(xyz, idx)
    assert gathered.shape == (2, 16, 3)
