import numpy as np
import torch


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    Center the point cloud at the origin and scale it to fit in the unit sphere.

    Args:
        points: numpy array of shape (N, 3)

    Returns:
        Normalized numpy array of shape (N, 3)
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid

    max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if max_dist > 0:
        points = points / max_dist

    return points


def sample_points(
    points: np.ndarray,
    labels: np.ndarray,
    num_points: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a fixed number of points and keep labels aligned.

    Args:
        points: numpy array of shape (N, 3)
        labels: numpy array of shape (N,)
        num_points: number of points to sample

    Returns:
        sampled_points: shape (num_points, 3)
        sampled_labels: shape (num_points,)
    """
    n = len(points)

    if n >= num_points:
        idx = np.random.choice(n, num_points, replace=False)
    else:
        idx = np.random.choice(n, num_points, replace=True)

    return points[idx], labels[idx]


def sample_points_with_normals(
    points: np.ndarray,
    normals: np.ndarray,
    labels: np.ndarray,
    num_points: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a fixed number of points and keep normals/labels aligned.

    Args:
        points: numpy array of shape (N, 3)
        normals: numpy array of shape (N, 3)
        labels: numpy array of shape (N,)
        num_points: number of points to sample

    Returns:
        sampled_points: shape (num_points, 3)
        sampled_normals: shape (num_points, 3)
        sampled_labels: shape (num_points,)
    """
    n = len(points)

    if n >= num_points:
        idx = np.random.choice(n, num_points, replace=False)
    else:
        idx = np.random.choice(n, num_points, replace=True)

    return points[idx], normals[idx], labels[idx]


def to_tensor_points(points: np.ndarray) -> torch.Tensor:
    """
    Convert point features to a PyTorch float tensor.

    Args:
        points: numpy array of point features, typically shape (N, 3) or (N, 6)

    Returns:
        PyTorch tensor of type float32
    """
    return torch.tensor(points, dtype=torch.float32)


def to_tensor_labels(labels: np.ndarray) -> torch.Tensor:
    """
    Convert labels to a PyTorch long tensor.

    Args:
        labels: numpy array of labels, typically shape (N,)

    Returns:
        PyTorch tensor of type long
    """
    return torch.tensor(labels, dtype=torch.long)