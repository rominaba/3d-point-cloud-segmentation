import pytest
import torch

from src.pointnetpp.classification import PointNetPPClassifier


@pytest.mark.parametrize("in_channels", [3, 6])
def test_pointnetpp_classifier_forward_shape(in_channels: int):
    b, n = 2, 64
    num_classes = 5
    x = torch.randn(b, n, in_channels)

    model = PointNetPPClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        sa1_npoint=16,
        sa1_coarse_npoint=8,
        sa2_npoint=4,
        sa2_coarse_npoint=2,
        sa1_fused_dim=32,
        sa2_fused_dim=64,
        head_mlp_dims=[64, 32],
    )
    logits = model(x)
    assert logits.shape == (b, num_classes)
