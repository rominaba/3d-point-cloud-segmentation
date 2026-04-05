from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, DATA_ROOT, NUM_POINTS, USE_NORMALS
from src.dataset import ShapeNetPartDataset, load_category_mapping, load_splits
from src.pointnetpp.classification import PointNetPPClassifier
from src.utils.utils import choose_device, collect_garbage, set_seed
from src.utils.utils import get_logger
from datetime import datetime
logger = get_logger("train_pointnetpp_classification", write_to_file=True)

# Set seed for reproducibility
set_seed()
current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for points, class_labels, _seg_labels in loader:
            points = points.to(device)
            class_labels = class_labels.to(device)
            logits = model(points)
            loss = loss_fn(logits, class_labels)
            total_loss += loss.item() * points.size(0)
            pred = logits.argmax(dim=1)
            total_correct += (pred == class_labels).sum().item()
            total += points.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PointNet++ classifier on ShapeNetPart category labels.")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--use-normals", action="store_true", default=USE_NORMALS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--sa-aggregation",
        type=str,
        choices=("mrg", "msg"),
        default="msg",
        help="Set abstraction local features grouping method: mrg (multiresolution) or msg (multiscale).",
    )
    args = parser.parse_args()
    sa_aggregation = {"mrg": "multiresolution", "msg": "multiscale"}[args.sa_aggregation]
    logger.info(f"Training PointNet++ classifier on ShapeNetPart category labels with {sa_aggregation} aggregation method.")
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
    num_classes = len(train_dataset.category_to_idx)
    model = PointNetPPClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        sa_aggregation=sa_aggregation,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"pointnetpp-classification-with-{sa_aggregation}-{current_time}.pt"
    for epoch in range(1, args.epochs + 1):
        collect_garbage(device)
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for points, class_labels, _seg_labels in train_loader:
            points = points.to(device)
            class_labels = class_labels.to(device)

            optimizer.zero_grad()
            logits = model(points)
            loss = loss_fn(logits, class_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * points.size(0)
            running_correct += (logits.argmax(dim=1) == class_labels).sum().item()
            total += points.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = running_correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_channels": in_channels,
                    "num_classes": num_classes,
                    "use_normals": args.use_normals,
                    "sa_aggregation": sa_aggregation,
                },
                save_path,
            )
            logger.info(f"Saved best checkpoint to {save_path}")


if __name__ == "__main__":
    main()

