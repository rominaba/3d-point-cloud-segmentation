import pytest
import torch

from src.pointnetpp.part_segmentation import PointNetPPPartSeg


@pytest.mark.parametrize("in_channels", [3, 6])
def test_pointnetpp_partseg_forward_shapes(in_channels: int):
    torch.manual_seed(0)
    b, n = 2, 64
    num_parts = 10
    x = torch.randn(b, n, in_channels)

    model = PointNetPPPartSeg(
        in_channels=in_channels,
        num_part_classes=num_parts,
        # Make it small enough for unit tests.
        sa1_npoint=16,
        sa1_coarse_npoint=8,
        sa2_npoint=4,
        sa2_coarse_npoint=2,
        sa1_fused_dim=32,
        sa2_fused_dim=64,
        fp2_mlp_dims=[64, 32],
        fp1_mlp_dims=[32, 32],
        head_mlp_dims=[32, 16],
        fp_k=3,
    )

    logits = model(x)
    assert logits.shape == (b, n, num_parts)


def test_pointnetpp_partseg_with_category_embedding():
    torch.manual_seed(0)
    b, n, c = 2, 48, 3
    num_parts = 8
    num_categories = 3

    x = torch.randn(b, n, c)
    class_label = torch.tensor([0, 2], dtype=torch.long)

    model = PointNetPPPartSeg(
        in_channels=c,
        num_part_classes=num_parts,
        num_categories=num_categories,
        category_embed_dim=4,
        sa1_npoint=12,
        sa1_coarse_npoint=6,
        sa2_npoint=3,
        sa2_coarse_npoint=2,
        sa1_fused_dim=16,
        sa2_fused_dim=32,
        fp2_mlp_dims=[32, 16],
        fp1_mlp_dims=[16, 16],
        head_mlp_dims=[16, 8],
    )

    logits = model(x, class_label=class_label)
    assert logits.shape == (b, n, num_parts)
