from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import statistics

import torch

from src.visualize import visualize_part_seg_comparison

SEG_CLASSES = {
    "Airplane": [0, 1, 2, 3],
    "Bag": [4, 5],
    "Cap": [6, 7],
    "Car": [8, 9, 10, 11],
    "Chair": [12, 13, 14, 15],
    "Earphone": [16, 17, 18],
    "Guitar": [19, 20, 21],
    "Knife": [22, 23],
    "Lamp": [24, 25, 26, 27],
    "Laptop": [28, 29],
    "Motorbike": [30, 31, 32, 33, 34, 35],
    "Mug": [36, 37],
    "Pistol": [38, 39, 40],
    "Rocket": [41, 42, 43],
    "Skateboard": [44, 45, 46],
    "Table": [47, 48, 49],
}

def compute_instance_miou(pred_labels, true_labels, part_ids):
    """
    Compute mIoU for one object instance.

    Args:
        pred_labels: (N,) predicted part labels
        true_labels: (N,) ground-truth part labels
        part_ids: valid part ids for this shape category

    Returns:
        float mIoU for this instance
    """
    part_ious = []
    for part_id in part_ids:
        pred_mask = pred_labels == part_id
        true_mask = true_labels == part_id

        union = (pred_mask | true_mask).sum()
        if union == 0:
            iou = 1.0
        else:
            intersection = (pred_mask & true_mask).sum()
            iou = float(intersection) / float(union)

        part_ious.append(iou)

    return sum(part_ious) / len(part_ious)


@torch.no_grad()
def evaluate_partseg(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    class_idx_to_cat: dict[int, str],
    cat_to_parts: dict[str, list[int]] | None = None,
    loss_fn=None,
    use_category_conditioning: bool = False,
    save_visualizations: bool = False,
    visualize_dir: str = "visuals",
    num_votes: int = 1,
    vote_jitter_std: float = 0.0,
) -> dict:
    """
    Evaluate part segmentation model.

    Expected loader batch format:
        points, class_labels, seg_labels
    where
        points: (B, N, C)
        class_labels: (B,)
        seg_labels: (B, N)

    Args:
        model: segmentation model
        loader: dataloader
        device: torch.device
        class_idx_to_cat: maps object class index to category name
        cat_to_parts: category to valid part ids
        loss_fn: optional loss function
        use_category_conditioning: whether model expects class labels too
        save_visualizations: if True, save one comparison PNG per object category
        visualize_dir: output directory when save_visualizations is True
        num_votes: test-time augmentation — run forward this many times and average
            logits before argmax (same protocol as yanx27 Pointnet2 test_partseg).
        vote_jitter_std: if > 0, add Gaussian noise (std on XYZ) for votes after the
            first so averaged logits differ under deterministic models;0 matches the
            reference repo (repeated identical forwards).

    Returns:
        dict with evaluation metrics, including:
        - accuracy_std_over_instances: sample std of point accuracy per object (spread
          across shapes in the evaluated split).
        - accuracy_std_over_batches: sample std of mean point accuracy per batch
          (spread across minibatches; depends on batch size and ordering).
        - per_vote_accuracy: list of length num_votes — overall point accuracy if
          only that vote were used (argmax on that pass's logits, no averaging).
        - accuracy_std_over_votes: sample std of per_vote_accuracy (spread across
          voting passes; 0 when num_votes < 2 or votes are identical).
    """
    if cat_to_parts is None:
        cat_to_parts = SEG_CLASSES

    model.eval()

    num_part_classes = max(part_id for parts in cat_to_parts.values() for part_id in parts) + 1

    total_correct = 0
    total_seen = 0

    total_correct_class = [0 for _ in range(num_part_classes)]
    total_seen_class = [0 for _ in range(num_part_classes)]

    shape_ious: dict[str, list[float]] = defaultdict(list)

    total_loss = 0.0
    num_batches = 0

    batch_point_accs: list[float] = []
    instance_point_accs: list[float] = []

    vote_point_correct = [0 for _ in range(num_votes)]
    vote_point_seen = [0 for _ in range(num_votes)]

    saved_viz_categories: set[str] | None = set() if save_visualizations else None
    if save_visualizations:
        Path(visualize_dir).mkdir(parents=True, exist_ok=True)
    num_categories_for_viz = len(class_idx_to_cat)

    for batch in loader:
        if len(batch) != 3:
            raise ValueError(
                "Expected batch to have 3 items: (points, class_labels, seg_labels)"
            )

        points, class_labels, seg_labels = batch
        points = points.to(device)
        class_labels = class_labels.to(device)
        seg_labels = seg_labels.to(device)

        vote_pool: torch.Tensor | None = None
        for v in range(num_votes):
            pts = points
            if vote_jitter_std > 0.0 and v > 0:
                pts = points.clone()
                noise = torch.randn_like(pts[..., :3], device=pts.device, dtype=pts.dtype)
                pts[..., :3] = pts[..., :3] + noise * vote_jitter_std
            if use_category_conditioning:
                logits_v = model(pts, class_labels)
            else:
                logits_v = model(pts)
            logits_v = logits_v.float()
            pred_v = logits_v.argmax(dim=-1)
            vote_point_correct[v] += (pred_v == seg_labels).sum().item()
            vote_point_seen[v] += seg_labels.numel()
            vote_pool = logits_v if vote_pool is None else vote_pool + logits_v

        assert vote_pool is not None
        logits = vote_pool / float(num_votes)

        if logits.dim() != 3:
            raise ValueError(f"Expected logits shape (B, N, num_parts), got {tuple(logits.shape)}")

        pred = logits.argmax(dim=-1)

        if saved_viz_categories is not None and len(saved_viz_categories) < num_categories_for_viz:
            out_dir = Path(visualize_dir)
            for b in range(points.size(0)):
                if len(saved_viz_categories) >= num_categories_for_viz:
                    break
                cat_idx = int(class_labels[b].item())
                cat_name = class_idx_to_cat[cat_idx]
                if cat_name in saved_viz_categories:
                    continue
                pts = points[b].detach().cpu().numpy()[:, :3]
                pred_np = pred[b].detach().cpu().numpy()
                gt_np = seg_labels[b].detach().cpu().numpy()
                visualize_part_seg_comparison(
                    pts,
                    pred_np,
                    gt_np,
                    title=cat_name,
                    save_path=str(out_dir / f"{cat_name}_example.png"),
                )
                saved_viz_categories.add(cat_name)

        if loss_fn is not None:
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), seg_labels.reshape(-1))
            total_loss += loss.item()

        num_batches += 1

        batch_correct = (pred == seg_labels).sum().item()
        batch_seen = seg_labels.numel()
        if batch_seen > 0:
            batch_point_accs.append(batch_correct / float(batch_seen))

        # update overall accuracy
        total_correct += batch_correct
        total_seen += batch_seen

        for i in range(points.size(0)):
            inst_correct = (pred[i] == seg_labels[i]).sum().item()
            inst_n = int(seg_labels[i].numel())
            if inst_n > 0:
                instance_point_accs.append(inst_correct / float(inst_n))

        # update class avg accuracy totals
        for part_id in range(num_part_classes):
            gt_mask = seg_labels == part_id
            total_seen_class[part_id] += gt_mask.sum().item()
            total_correct_class[part_id] += ((pred == part_id) & gt_mask).sum().item()

        # compute per-instance mIoU
        batch_size = points.size(0)
        for i in range(batch_size):
            class_idx = int(class_labels[i].item())
            cat = class_idx_to_cat[class_idx]
            part_ids = cat_to_parts[cat]

            pred_i = pred[i]
            true_i = seg_labels[i]

            instance_miou = compute_instance_miou(pred_i, true_i, part_ids)
            shape_ious[cat].append(instance_miou)

    accuracy = total_correct / total_seen if total_seen > 0 else 0.0

    class_accs: list[float] = []
    for correct, seen in zip(total_correct_class, total_seen_class):
        if seen > 0:
            class_accs.append(correct / seen)
    class_avg_accuracy = sum(class_accs) / len(class_accs) if class_accs else 0.0

    per_category_miou = {
        cat: (sum(ious) / len(ious) if len(ious) > 0 else 0.0)
        for cat, ious in shape_ious.items()
    }

    class_avg_miou = (
        sum(per_category_miou.values()) / len(per_category_miou)
        if len(per_category_miou) > 0
        else 0.0
    )

    all_instance_ious = [iou for ious in shape_ious.values() for iou in ious]
    instance_avg_miou = (
        sum(all_instance_ious) / len(all_instance_ious)
        if len(all_instance_ious) > 0
        else 0.0
    )

    avg_loss = total_loss / num_batches if (loss_fn is not None and num_batches > 0) else None

    def _stdev(xs: list[float]) -> float:
        if len(xs) < 2:
            return 0.0
        return float(statistics.stdev(xs))

    accuracy_std_over_instances = _stdev(instance_point_accs)
    accuracy_std_over_batches = _stdev(batch_point_accs)

    per_vote_accuracy: list[float] = []
    for v in range(num_votes):
        vs = vote_point_seen[v]
        per_vote_accuracy.append(
            float(vote_point_correct[v]) / float(vs) if vs > 0 else 0.0
        )
    accuracy_std_over_votes = _stdev(per_vote_accuracy) if num_votes > 1 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "per_vote_accuracy": per_vote_accuracy,
        "accuracy_std_over_votes": accuracy_std_over_votes,
        "accuracy_std_over_instances": accuracy_std_over_instances,
        "accuracy_std_over_batches": accuracy_std_over_batches,
        "class_avg_accuracy": class_avg_accuracy,
        "class_avg_miou": class_avg_miou,
        "instance_avg_miou": instance_avg_miou,
        "per_category_miou": dict(sorted(per_category_miou.items())),
    }