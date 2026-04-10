from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, DATA_ROOT, NUM_POINTS, USE_NORMALS
from src.dataset import ShapeNetPartDataset, load_category_mapping, load_splits
from src.pointnetpp.part_segmentation import PointNetPPPartSeg
from src.utils.utils import get_logger, choose_device, set_seed
from src.utils.partseg_metrics import evaluate_partseg, SEG_CLASSES

set_seed()
logger = get_logger("test_pointnetpp_part_segmentation", write_to_file=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="Test PointNet++ part segmentation on ShapeNetPart test split.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--use-normals", action="store_true", default=USE_NORMALS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-part-classes", type=int, default=50)
    parser.add_argument("--use-category-conditioning", action="store_true")
    parser.add_argument("--category-embed-dim", type=int, default=16)
    parser.add_argument(
        "--sa-aggregation",
        type=str,
        choices=("mrg", "msg"),
        default="mrg",
        help="Set abstraction local features: MRG (multiresolution) or MSG (multiscale).",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip saving one comparison PNG per object category under --visualize-dir.",
    )
    parser.add_argument(
        "--visualize-dir",
        type=str,
        default="visuals",
        help="Directory for part-segmentation comparison images (when visualization is enabled).",
    )
    parser.add_argument(
        "--num-votes",
        type=int,
        default=3,
        help="Test-time votes: average logits over this many forwards (yanx27 test_partseg default: 3).",
    )
    parser.add_argument(
        "--vote-jitter-std",
        type=float,
        default=0.01,
        help="Standard deviation of Gaussian noise on XYZ for votes after the first;0 repeats identical passes.",
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

    use_category_conditioning = bool(
        ckpt.get("use_category_conditioning", args.use_category_conditioning)
    )

    # Training saves num_categories / category_embed_dim even when the model was built
    # without category conditioning; only enable embedding when weights match.
    ckpt_skip = frozenset(
        {
            "model_state_dict",
            "use_normals",
            "use_category_conditioning",
            "pointnetpp_partseg_kwargs",
        }
    )
    model_kwargs = {k: v for k, v in ckpt.items() if k not in ckpt_skip}
    if use_category_conditioning:
        model_kwargs["num_categories"] = int(model_kwargs["num_categories"])
        model_kwargs["category_embed_dim"] = int(
            model_kwargs.get("category_embed_dim", args.category_embed_dim)
        )
    else:
        model_kwargs["num_categories"] = None
        model_kwargs["category_embed_dim"] = 0

    logger.info(f"Testing {args.checkpoint}...")
    logger.info(
        f"Set Abstraction Grouping Method: {ckpt['sa_aggregation']}"
        f"use_category_conditioning={use_category_conditioning}"
    )
    logger.info(
        f"Test-time voting: num_votes={args.num_votes}, vote_jitter_std={args.vote_jitter_std}"
    )
    model = PointNetPPPartSeg(**model_kwargs).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    class_idx_to_cat = test_dataset.idx_to_category

    metrics = evaluate_partseg(
        model=model,
        loader=test_loader,
        device=device,
        class_idx_to_cat=class_idx_to_cat,
        cat_to_parts=SEG_CLASSES,
        loss_fn=None,
        use_category_conditioning=use_category_conditioning,
        save_visualizations=not args.no_visualization,
        visualize_dir=args.visualize_dir,
        num_votes=args.num_votes,
        vote_jitter_std=args.vote_jitter_std,
    )

    for cat, miou in metrics["per_category_miou"].items():
        logger.info(f"eval mIoU of {cat:<15} {miou:.6f}")

    logger.info(f"Accuracy is: {metrics['accuracy']:.5f}")
    if args.num_votes > 1:
        logger.info(f"Per-vote accuracies: {metrics['per_vote_accuracy']}")
        logger.info(
            f"Accuracy standard deviation (over votes): {metrics['accuracy_std_over_votes']:.5f}"
        )
    logger.info(
        f"Accuracy standard deviation (over instances): {metrics['accuracy_std_over_instances']:.5f}"
    )
    logger.info(
        f"Accuracy standard deviation (over batches): {metrics['accuracy_std_over_batches']:.5f}"
    )
    logger.info(f"Class avg accuracy is: {metrics['class_avg_accuracy']:.5f}")
    logger.info(f"Class avg mIOU is: {metrics['class_avg_miou']:.5f}")
    logger.info(f"Instance avg mIOU is: {metrics['instance_avg_miou']:.5f}")


if __name__ == "__main__":
    main()

