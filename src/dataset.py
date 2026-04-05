import json
import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    normalize_point_cloud,
    sample_points,
    sample_points_with_normals,
)


def load_category_mapping(data_root: str) -> dict[str, str]:
    """
    Load the mapping from category name to synset folder id.

    Returns:
        Example:
        {
            "Airplane": "02691156",
            "Chair": "03001627",
            ...
        }
    """
    mapping_path = os.path.join(data_root, "synsetoffset2category.txt")
    category_mapping = {}

    with open(mapping_path, "r") as f:
        for line in f:
            name, folder = line.strip().split()
            category_mapping[name] = folder

    return category_mapping


def load_splits(data_root: str) -> tuple[list[str], list[str], list[str]]:
    """
    Load official train/val/test split lists.
    """
    split_dir = os.path.join(data_root, "train_test_split")

    with open(os.path.join(split_dir, "shuffled_train_file_list.json"), "r") as f:
        train_list = json.load(f)

    with open(os.path.join(split_dir, "shuffled_val_file_list.json"), "r") as f:
        val_list = json.load(f)

    with open(os.path.join(split_dir, "shuffled_test_file_list.json"), "r") as f:
        test_list = json.load(f)

    return train_list, val_list, test_list


def get_sample_details(
    split_list: list[str],
    idx: int,
    data_root: str,
) -> tuple[str, list[str], str, str, str]:
    """
    Resolve a split entry into category id, file id, and full file path.
    """
    entry = split_list[idx]
    parts = entry.split("/")
    category_id = parts[-2]
    file_id = parts[-1]
    sample_path = os.path.join(data_root, category_id, file_id + ".txt")

    return entry, parts, category_id, file_id, sample_path


def load_raw_sample(sample_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one raw ShapeNet Part sample.

    Returns:
        xyz: shape (N, 3)
        normals: shape (N, 3)
        seg_labels: shape (N,)
    """
    arr = np.loadtxt(sample_path)
    xyz = arr[:, 0:3]
    normals = arr[:, 3:6]
    seg_labels = arr[:, 6].astype(int)

    return xyz, normals, seg_labels


def build_class_mappings(
    category_mapping: dict[str, str],
) -> tuple[dict[str, int], dict[str, int], dict[int, str], dict[str, str]]:
    """
    Build mappings between category names, folder ids, and class indices.
    """
    category_to_idx = {
        name: i for i, name in enumerate(sorted(category_mapping.keys()))
    }
    folder_to_class_idx = {
        folder: category_to_idx[name] for name, folder in category_mapping.items()
    }
    class_idx_to_name = {
        i: name for i, name in enumerate(sorted(category_mapping.keys()))
    }
    reverse_mapping = {folder: name for name, folder in category_mapping.items()}

    return category_to_idx, folder_to_class_idx, class_idx_to_name, reverse_mapping


class ShapeNetPartDataset(Dataset):
    """
    PyTorch dataset for ShapeNet Part segmentation.
    """

    def __init__(
        self,
        data_root: str,
        split_list: list[str],
        category_mapping: dict[str, str],
        num_points: int = 1024,
        use_normals: bool = False,
    ) -> None:
        self.data_root = data_root
        self.split_list = split_list
        self.category_mapping = category_mapping
        self.num_points = num_points
        self.use_normals = use_normals

        (
            self.category_to_idx,
            self.folder_to_class_idx,
            self.class_idx_to_name,
            self.reverse_mapping,
        ) = build_class_mappings(category_mapping)
        
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}

    def __len__(self) -> int:
        return len(self.split_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, category_id, _, sample_path = get_sample_details(
            self.split_list, idx, self.data_root
        )

        xyz, normals, seg_labels = load_raw_sample(sample_path)
        xyz = normalize_point_cloud(xyz)

        if self.use_normals:
            xyz, normals, seg_labels = sample_points_with_normals(
                xyz, normals, seg_labels, num_points=self.num_points
            )
            features = np.concatenate([xyz, normals], axis=1)
        else:
            features, seg_labels = sample_points(
                xyz, seg_labels, num_points=self.num_points
            )

        class_label = self.folder_to_class_idx[category_id]

        features = torch.tensor(features, dtype=torch.float32)
        class_label = torch.tensor(class_label, dtype=torch.long)
        seg_labels = torch.tensor(seg_labels, dtype=torch.long)

        return features, class_label, seg_labels