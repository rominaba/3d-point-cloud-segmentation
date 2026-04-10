from __future__ import annotations

import argparse
import statistics

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
    parser.add_argument(
        "--num-votes",
        type=int,
        default=3,
        help="Average logits over this many forwards (use 3 to mirror PointNet++ test voting).",
    )
    parser.add_argument(
        "--vote-jitter-std",
        type=float,
        default=0.02,
        help="Gaussian noise std on XYZ for votes after the first;0 = identical passes.",
    )
    
    args = parser.parse_args()

    device = choose_device(logger)
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    use_normals = bool(ckpt.get("use_normals", args.use_normals))

    category_mapping = load_category_mapping(args.data_root)
    _train_list, _val_list, test_list = load_splits(args.data_root)
    test_dataset = ShapeNetPartDataset(
        data_root=args.data_root,
        split_list=test_list,
        category_mapping=category_mapping,
        num_points=args.num_points,
        use_normals=use_normals,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    ckpt_skip = frozenset(
        {
            "model_state_dict",
            "use_normals",
        }
    )
    model_kwargs = {k: v for k, v in ckpt.items() if k not in ckpt_skip}
    
    model_kwargs["in_channels"] = int(model_kwargs["in_channels"])
    model_kwargs["num_classes"] = int(model_kwargs["num_classes"])

    logger.info(f"Testing {args.checkpoint}...")
    logger.info(f"Set Abstraction Grouping Method: {model_kwargs['sa_aggregation']}")
    logger.info(
        f"Test-time voting: num_votes={args.num_votes}, vote_jitter_std={args.vote_jitter_std}"
    )
    model = PointNetPPClassifier(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    vote_correct = [0 for _ in range(args.num_votes)]
    total = 0
    total_correct = 0
    with torch.no_grad():
        for points, class_labels, _seg_labels in tqdm(
            test_loader, desc="Predicting", unit="batch"
        ):
            points = points.to(device)
            class_labels = class_labels.to(device)
            vote_pool: torch.Tensor | None = None
            for v in range(args.num_votes):
                pts = points
                if args.vote_jitter_std > 0.0 and v > 0:
                    pts = points.clone()
                    noise = torch.randn_like(pts[..., :3], device=pts.device, dtype=pts.dtype)
                    pts[..., :3] = pts[..., :3] + noise * args.vote_jitter_std
                logits_v = model(pts).float()
                pred_v = logits_v.argmax(dim=1)
                vote_correct[v] += (pred_v == class_labels).sum().item()
                vote_pool = logits_v if vote_pool is None else vote_pool + logits_v
            assert vote_pool is not None
            logits = vote_pool / float(args.num_votes)
            pred = logits.argmax(dim=1)
            total_correct += (pred == class_labels).sum().item()
            total += points.size(0)

    acc = total_correct / max(total, 1)
    per_vote_acc = [vote_correct[v] / max(total, 1) for v in range(args.num_votes)]
    acc_std_over_votes = float(statistics.stdev(per_vote_acc)) if args.num_votes > 1 else 0.0
    logger.info(f"Test classification accuracy (averaged logits): {acc:.4f}")
    if args.num_votes > 1:
        logger.info(f"Per-vote accuracies: {per_vote_acc}")
        logger.info(f"Accuracy standard deviation (over votes): {acc_std_over_votes:.4f}")


if __name__ == "__main__":
    main()

