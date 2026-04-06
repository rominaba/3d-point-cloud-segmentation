from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import BATCH_SIZE, DATA_ROOT, NUM_POINTS, USE_NORMALS
from src.dataset import ShapeNetPartDataset, load_category_mapping, load_splits
from src.pointnetpp.classification import PointNetPPClassifier
from src.utils.utils import get_logger, choose_device

logger = get_logger("test_pointnetpp_classification", write_to_file=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Test PointNet++ classifier on ShapeNetPart test split.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--use-normals", action="store_true", default=USE_NORMALS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
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
    num_classes = int(ckpt["num_classes"])
    sa_aggregation = ckpt.get("sa_aggregation")
    logger.info(f"Testing {args.checkpoint}...")
    logger.info(f"Set Abstraction Grouping Method: {sa_aggregation}")
    model = PointNetPPClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        sa_aggregation=sa_aggregation,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total = 0
    total_correct = 0
    with torch.no_grad():
        for points, class_labels, _seg_labels in tqdm(
            test_loader, desc="Predicting", unit="batch"
        ):
            points = points.to(device)
            class_labels = class_labels.to(device)
            logits = model(points)
            pred = logits.argmax(dim=1)
            total_correct += (pred == class_labels).sum().item()
            total += points.size(0)

    acc = total_correct / max(total, 1)
    logger.info(f"Test classification accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()

