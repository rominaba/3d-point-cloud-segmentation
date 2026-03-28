from torch.utils.data import DataLoader

from src.config import DATA_ROOT, NUM_POINTS, BATCH_SIZE, USE_NORMALS
from src.dataset import load_category_mapping, load_splits, ShapeNetPartDataset


def main() -> None:
    category_mapping = load_category_mapping(DATA_ROOT)
    train_list, val_list, test_list = load_splits(DATA_ROOT)

    train_dataset = ShapeNetPartDataset(
        data_root=DATA_ROOT,
        split_list=train_list,
        category_mapping=category_mapping,
        num_points=NUM_POINTS,
        use_normals=USE_NORMALS,
    )

    val_dataset = ShapeNetPartDataset(
        data_root=DATA_ROOT,
        split_list=val_list,
        category_mapping=category_mapping,
        num_points=NUM_POINTS,
        use_normals=USE_NORMALS,
    )

    test_dataset = ShapeNetPartDataset(
        data_root=DATA_ROOT,
        split_list=test_list,
        category_mapping=category_mapping,
        num_points=NUM_POINTS,
        use_normals=USE_NORMALS,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))

    batch_points, batch_class_labels, batch_seg_labels = next(iter(train_loader))

    print("Batch points shape:", batch_points.shape)
    print("Batch class labels shape:", batch_class_labels.shape)
    print("Batch seg labels shape:", batch_seg_labels.shape)


if __name__ == "__main__":
    main()