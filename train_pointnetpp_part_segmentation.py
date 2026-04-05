from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, DATA_ROOT, NUM_POINTS, USE_NORMALS
from src.dataset import ShapeNetPartDataset, load_category_mapping, load_splits
from src.pointnetpp.part_segmentation import PointNetPPPartSeg
from src.utils.utils import get_logger, choose_device, set_seed, collect_garbage
from src.utils.partseg_metrics import evaluate_partseg, SEG_CLASSES

current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
set_seed()
logger = get_logger("train_pointnetpp_part_segmentation", write_to_file=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PointNet++ part segmentation on ShapeNetPart.")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--use-normals", action="store_true", default=USE_NORMALS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-part-classes", type=int, default=50)
    parser.add_argument("--use-category-conditioning", action="store_true")
    parser.add_argument("--category-embed-dim", type=int, default=16)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--sa-aggregation",
        type=str,
        choices=("mrg", "msg"),
        default="mrg",
        help="Set abstraction local features: MRG (multiresolution) or MSG (multiscale).",
    )
    args = parser.parse_args()
    sa_aggregation = {"mrg": "multiresolution", "msg": "multiscale"}[args.sa_aggregation]
    logger.info(f"Training PointNet++ part segmentation on ShapeNetPart with {sa_aggregation} aggregation method.")
    device = choose_device(logger)

    category_mapping = load_category_mapping(args.data_root)
    train_list, val_list, _test_list = load_splits(args.data_root)

    train_dataset = ShapeNetPartDataset(
        data_root=args.data_root,
        split_list=train_list,
        category_mapping=category_mapping,
        num_points=args.num_points,
        use_normals=args.use_normals,
    )
    val_dataset = ShapeNetPartDataset(
        data_root=args.data_root,
        split_list=val_list,
        category_mapping=category_mapping,
        num_points=args.num_points,
        use_normals=args.use_normals,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    in_channels = 6 if args.use_normals else 3
    model = PointNetPPPartSeg(
        in_channels=in_channels,
        num_part_classes=args.num_part_classes,
        num_categories=(len(train_dataset.category_to_idx) if args.use_category_conditioning else None),
        category_embed_dim=(args.category_embed_dim if args.use_category_conditioning else 0),
        sa_aggregation=sa_aggregation,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    class_idx_to_cat = train_dataset.idx_to_category
    
    best_accuracy = -1.0
    best_class_avg_miou = -1.0
    best_instance_avg_miou = -1.0

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"pointnetpp-part-segmentation-with-{sa_aggregation}-{current_time}.pt"

    for epoch in range(1, args.epochs + 1):
        collect_garbage(device)
        model.train()

        running_loss = 0.0
        running_correct = 0
        total_points = 0

        for points, class_labels, seg_labels in train_loader:
            points = points.to(device)
            class_labels = class_labels.to(device)
            seg_labels = seg_labels.to(device)

            optimizer.zero_grad()
            logits = (
                model(points, class_labels)
                if args.use_category_conditioning
                else model(points)
            )

            loss = loss_fn(logits.reshape(-1, logits.size(-1)), seg_labels.reshape(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * points.size(0)

            pred = logits.argmax(dim=-1)
            running_correct += (pred == seg_labels).sum().item()
            total_points += seg_labels.numel()

        train_loss = running_loss / max(len(train_loader.dataset), 1)
        train_point_acc = running_correct / max(total_points, 1)

        metrics = evaluate_partseg(
            model=model,
            loader=val_loader,
            device=device,
            class_idx_to_cat=class_idx_to_cat,
            cat_to_parts=SEG_CLASSES,
            loss_fn=loss_fn,
            use_category_conditioning=args.use_category_conditioning,
        )

        logger.info(f"Epoch {epoch} ({epoch}/{args.epochs}):")
        logger.info(f"Train loss is: {train_loss:.5f}")
        logger.info(f"Train accuracy is: {train_point_acc:.5f}")

        for cat, miou in metrics["per_category_miou"].items():
            logger.info(f"eval mIoU of {cat:<15} {miou:.6f}")

        logger.info(
            f"Epoch {epoch} test Accuracy: {metrics['accuracy']:.6f}  "
            f"Class avg mIOU: {metrics['class_avg_miou']:.6f}   "
            f"Instance avg mIOU: {metrics['instance_avg_miou']:.6f}"
        )

        if metrics["loss"] is not None:
            logger.info(f"Validation loss is: {metrics['loss']:.5f}")

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "num_part_classes": args.num_part_classes,
                    "use_normals": args.use_normals,
                    "use_category_conditioning": args.use_category_conditioning,
                    "num_categories": len(train_dataset.category_to_idx),
                    "category_embed_dim": args.category_embed_dim,
                    "sa_aggregation": sa_aggregation,
                },
                save_path,
            )
            logger.info(f"Saved best checkpoint to {save_path}")

        best_class_avg_miou = max(best_class_avg_miou, metrics["class_avg_miou"])
        best_instance_avg_miou = max(best_instance_avg_miou, metrics["instance_avg_miou"])

        logger.info(f"Best accuracy is: {best_accuracy:.5f}")
        logger.info(f"Best class avg mIOU is: {best_class_avg_miou:.5f}")
        logger.info(f"Best instance avg mIOU is: {best_instance_avg_miou:.5f}")

    logger.info(f"Training complete. Best checkpoint saved at: {save_path}")


if __name__ == "__main__":
    main()

