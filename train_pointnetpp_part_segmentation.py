from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, DATA_ROOT, NUM_POINTS, USE_NORMALS
from src.dataset import ShapeNetPartDataset, load_category_mapping, load_splits
from src.pointnetpp.part_segmentation import PointNetPPPartSeg
from src.utils.utils import get_logger

logger = get_logger("train_pointnetpp_part_segmentation")

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_category_conditioning: bool,
) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_points = 0

    with torch.no_grad():
        for points, class_labels, seg_labels in loader:
            points = points.to(device)
            class_labels = class_labels.to(device)
            seg_labels = seg_labels.to(device)

            logits = (
                model(points, class_labels) if use_category_conditioning else model(points)
            )
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), seg_labels.reshape(-1))
            total_loss += loss.item() * points.size(0)

            pred = logits.argmax(dim=-1)
            total_correct += (pred == seg_labels).sum().item()
            total_points += seg_labels.numel()

    mean_loss = total_loss / max(len(loader.dataset), 1)
    point_acc = total_correct / max(total_points, 1)
    return mean_loss, point_acc


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
    parser.add_argument("--save-path", type=str, default="checkpoints/pointnetpp_part_seg.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
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
        val_loss, val_point_acc = evaluate(
            model, val_loader, device, args.use_category_conditioning
        )
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_point_acc={train_point_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_point_acc={val_point_acc:.4f}"
        )

        if val_point_acc > best_val_acc:
            best_val_acc = val_point_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "num_part_classes": args.num_part_classes,
                    "use_normals": args.use_normals,
                    "use_category_conditioning": args.use_category_conditioning,
                    "num_categories": len(train_dataset.category_to_idx),
                    "category_embed_dim": args.category_embed_dim,
                },
                save_path,
            )
            logger.info(f"Saved best checkpoint to {save_path}")


if __name__ == "__main__":
    main()

