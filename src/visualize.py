import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

COLOURS = ["#2d4fdb", "#b82b2b",  "#2ad340", "#a920c2", "#e6ca29", "#f59031", "#ed81c2", "#040304", "#746B74", "#14DCE3"] 

# Helper function
def plot_part_seg(ax: Axes3D, points: np.ndarray, labels: np.ndarray, label_to_colour: dict[np.int64, str]):
    """
    Helper function to plot a 3D point cloud with points colour-coded by part labels.

    Args:
        ax: matplotlib 3D axis object on which the point cloud is plotted
        points: numpy array of shape (N, 3), where the dim 3 represents the xyz coordinates
        labels: numpy array of shape (N,) containing part labels for each point
        label_to_colour: dictionary mapping each unique label to a colour string
    """

    # Extract unique labels
    unique_labels = np.unique(labels)
    categorized_points = {}

    # Organize points by their label
    for i in range(len(points)):
        label = labels[i]
        if label not in categorized_points:
            categorized_points[label] = []
        categorized_points[label].append(points[i])

    # Plot the points for each label with the corresponding colour
    for i, label in enumerate(unique_labels):
        pts = np.array(categorized_points[label])

        ax.scatter(
            pts[:, 0], 
            pts[:, 1], 
            pts[:, 2], 
            s=2, 
            c=label_to_colour[label], 
            alpha=0.6
        )

    # Plot without axes for cleaner output and set consistent angle
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def visualize_part_seg(points: np.ndarray, labels: np.ndarray, title: str | None = None, save_path: str | None = None,) -> None:
    """
    Plot the 3D point cloud and colour code the points based on the part labels provided.

    Args: 
        points: numpy array of shape (N, 3), where the dim 3 represents the xyz coordinates
        labels: numpy array of shape (N,) containing part labels (either predictions or ground truth) for each point
        title: string representing the title that should be given to the plot
        save_path: string representing path of directory and image name for saving the plot (eg. "vis/sample_1.png"). If None, 
                    then plot will just open in a matplotlib window and not be saved.

    Note that N is the number of points per point cloud (eg. 1024)
    """

    # Extract unique labels and map each label to a distinct colour
    all_labels = np.unique(labels)
    label_to_colour = {
        label: COLOURS[i % len(COLOURS)] for i, label in enumerate(all_labels)
    }

    # Initialize figure and axis objects
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the points
    plot_part_seg(ax, points, labels, label_to_colour)

    # Set the title
    if title is not None: 
        ax.set_title(title)

    # Adjust spacing
    plt.tight_layout()

    # If save_path is provided, save the output to that path 
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    # Otherwise simply display via matplotlib window
    else:
        plt.show()


def visualize_part_seg_comparison(points: np.ndarray, pred_labels: np.ndarray, true_labels: np.ndarray, title: str | None = None, save_path: str | None = None,) -> None:
    """
    Plot the 3D point cloud twice side-by-side and colour code the points based on the predicted part labels in the first plot and 
    ground truth part labels in the second plot.

    Args: 
        points: numpy array of shape (N, 3), where the dim 3 represents the xyz coordinates
        pred_labels: numpy array of shape (N,) containing predicted part labels for each point
        true_labels: numpy array of shape (N,) containing ground truth part labels for each point
        title: string representing the title that should be given to the plot. If None, no title is displayed.
        save_path: string representing path of directory and image name for saving the plot (eg. "vis/sample_1.png"). If None, 
                    then plot will just open in a matplotlib window and not be saved.

    Note that N is the number of points per point cloud (eg. 1024)
    """

    # Extract unique labels and map each label to a distinct colour 
    # (ensures parts correspond to same colour for prediction and ground truth)
    all_labels = np.unique(np.concatenate([pred_labels, true_labels]))
    label_to_colour = {
        label: COLOURS[i % len(COLOURS)] for i, label in enumerate(all_labels) 
    }
   
    # Initialize figure object
    fig = plt.figure(figsize=(8, 4))

    # Initialize axis object for prediction, plot the points, and set the plot title
    ax1 = fig.add_subplot(121, projection="3d")
    plot_part_seg(ax1, points, pred_labels, label_to_colour)
    ax1.set_title("Prediction")

    # Initialize axis object for ground truth, plot the points, and set the plot title
    ax2 = fig.add_subplot(122, projection="3d")
    plot_part_seg(ax2, points, true_labels, label_to_colour)
    ax2.set_title("Ground Truth")

    # Set overall title
    if title is not None: 
        fig.suptitle(title)

    # Adjust spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # If save_path is provided, save the output to that path 
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    # Otherwise simply display via matplotlib window
    else:
        plt.show()
