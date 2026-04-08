from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import BATCH_SIZE, DATA_ROOT, NUM_POINTS, USE_NORMALS
from src.dataset import ShapeNetPartDataset, load_category_mapping, load_splits
from src.pointnetpp.part_segmentation import PointNetPPPartSeg
from src.utils.utils import get_logger, choose_device
from src.visualize import visualize_part_seg_comparison

logger = get_logger("test_pointnetpp_part_segmentation", write_to_file=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Test PointNet++ part segmentation on ShapeNetPart test split.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--use-normals", action="store_true", default=USE_NORMALS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--sa-aggregation",
        type=str,
        choices=("mrg", "msg"),
        default="mrg",
        help="Set abstraction local features: MRG (multiresolution) or MSG (multiscale).",
    )
    args = parser.parse_args()

    device = choose_device(logger)

    category_mapping = load_category_mapping(args.data_root)
    _train_list, _val_list, test_list = load_splits(args.data_root)
    test_dataset = ShapeNetPartDataset(
        data_root=args.data_root,
        split_list=test_list,
        category_mapping=category_mapping,
        num_points=args.num_points,
        use_normals=args.use_normals,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    in_channels = int(ckpt["in_channels"])
    num_part_classes = int(ckpt["num_part_classes"])
    use_category_conditioning = bool(ckpt.get("use_category_conditioning", False))
    num_categories = int(ckpt.get("num_categories", len(test_dataset.category_to_idx)))
    category_embed_dim = int(ckpt.get("category_embed_dim", 16))

    sa_aggregation = ckpt.get(
        "sa_aggregation",
        {"mrg": "multiresolution", "msg": "multiscale"}[args.sa_aggregation],
    )
    
    logger.info(f"Testing {args.checkpoint}...")
    logger.info(f"Set Abstraction Grouping Method: {sa_aggregation}")

    model = PointNetPPPartSeg(
        in_channels=in_channels,
        num_part_classes=num_part_classes,
        num_categories=(num_categories if use_category_conditioning else None),
        category_embed_dim=(category_embed_dim if use_category_conditioning else 0),
        sa_aggregation=sa_aggregation,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    idx_to_category = {v: k for k, v in test_dataset.category_to_idx.items()}
    saved_categories_for_visualize = set()

    total_correct = 0
    total_points = 0
    with torch.no_grad():
        for points, class_labels, seg_labels in tqdm(
            test_loader, desc="Predicting", unit="batch"
        ):
            points = points.to(device)
            class_labels = class_labels.to(device)
            seg_labels = seg_labels.to(device)

            logits = (
                model(points, class_labels) if use_category_conditioning else model(points)
            )
            pred = logits.argmax(dim=-1)
            total_correct += (pred == seg_labels).sum().item()
            total_points += seg_labels.numel()

            # Visualization
            for b in range(points.shape[0]):
                cat_idx = class_labels[b].item()
                cat_name = idx_to_category[cat_idx]

                if cat_name not in saved_categories_for_visualize:
                    pts = points[b].detach().cpu().numpy()[:,:3]
                    pred_np = pred[b].detach().cpu().numpy()
                    gt_np = seg_labels[b].detach().cpu().numpy()

                    visualize_part_seg_comparison(pts, pred_np, gt_np, title=cat_name, save_path=f"visuals/{cat_name}_example.png")
                    saved_categories_for_visualize.add(cat_name)
                
                if len(saved_categories_for_visualize) == len(idx_to_category):
                    break

    point_acc = total_correct / max(total_points, 1)
    logger.info(f"Test part-seg point accuracy: {point_acc:.4f}")


if __name__ == "__main__":
    main()

