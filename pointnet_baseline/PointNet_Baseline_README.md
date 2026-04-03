# PointNet Baseline Setup

This file documents how the PointNet baseline was set up and trained for ShapeNet Part segmentation.

## Source Repository

The baseline implementation was cloned from:

`https://github.com/yanx27/Pointnet_Pointnet2_pytorch`

This repository was used because it already provides PyTorch implementations of both PointNet and PointNet++, so it was a practical way to train a benchmark baseline without implementing the model from scratch.

## Local Setup

This setup was tested locally on a **MacBook Pro with Apple M1 Pro with 16-core built in GPU** using **PyTorch MPS** acceleration. The existing project virtual environment was reused.

The model was trained on the **official ShapeNet Part dataset**, which was already stored locally and excluded from version control through `.gitignore`.

## Main Changes Made

A few small compatibility changes were needed to run the repository locally.

- Updated the hardcoded ShapeNet Part dataset paths to the correct local dataset path.
- Replaced CUDA-only code with device-aware code using `mps`, `cuda`, or `cpu`.
- Updated `to_categorical()` so one-hot label tensors are created on the correct device.
- Fixed identity matrix creation in `models/pointnet_utils.py` so it works on MPS instead of only CUDA.
- Replaced deprecated NumPy aliases such as `np.float` with modern equivalents like `float`.
- Updated `torch.load(...)` in `test_partseg.py` to work with newer PyTorch versions using `weights_only=False` and `map_location=device`.

## Basic Setup Steps

Clone the repository:

```bash
cd /path/to/project-parent-folder
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git pointnet-baseline-repo
cd pointnet-baseline-repo
```

Activate the existing virtual environment and install any missing packages:

```bash
source /path/to/venv/bin/activate
pip install torch torchvision torchaudio numpy h5py matplotlib tqdm
```

## One-Step Script

A helper script called `train_pointnet_baseline.sh` was also created to automate the setup, patching, training, and evaluation steps.

It applies the required compatibility fixes, runs PointNet training, and then runs evaluation in one command.

## Training Configuration

The initial PointNet baseline was trained using:

- **xyz coordinates only**
- **1024 sampled points**
- **batch size 20**
- **5 epochs**

These are the current default settings and may be adjusted later during further experiments.

## Run Everything at Once

The full setup, training, and evaluation pipeline can also be run with:

```bash
chmod +x train_pointnet_baseline.sh
./train_pointnet_baseline.sh
```
If needed, the script can also be run with custom arguments for the repository folder and dataset path.

## Alternative Separate Training Command
To run just the training after making all the other changes just once. You can run the following command.

A short test run can be started with:

```bash
python train_partseg.py --model pointnet_part_seg --npoint 1024 --batch_size 20 --epoch 5 --log_dir pointnet_partseg_xyz_1024_bs20_test
```

A longer run can be started with:

```bash
python train_partseg.py --model pointnet_part_seg --npoint 1024 --batch_size 20 --log_dir pointnet_partseg_xyz_1024_bs20
```

## Alternative Separate Testing Command

After training, evaluation can be run with:

```bash
python test_partseg.py --log_dir pointnet_partseg_xyz_1024_bs20_test --num_point 1024 --batch_size 20
```

<!-- ## Important Test Script Fix

In `test_partseg.py`, the checkpoint load line was changed from:

```python
checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
```

to:

```python
checkpoint = torch.load(
    str(experiment_dir) + '/checkpoints/best_model.pth',
    map_location=device,
    weights_only=False
)
```

This was needed because newer PyTorch versions changed the default loading behavior. -->

## Logged Results

The training logs for this setup will be linked in:

`pointnet_part_seg.txt`

The evaluation results will also be recorded in:

`eval.txt`

These files include the main reported metrics, such as accuracy and IoU.

# Explanation of metrics for PointNet part segmentation

Since PointNet was used here for **3D point cloud part segmentation**, the model predicts a **part label for every point** in an object. So the main metrics measure how well the model labels points and how well the predicted parts overlap the true parts.

## Training log

### Train accuracy

This is the proportion of points whose predicted part label matches the ground-truth part label during training.

$$
\text{Train Accuracy} = \frac{1}{B}\sum_{b=1}^{B}
\frac{\text{num correct predictions in batch } b}{\text{total number of points evaluated in batch } b}
$$

where

$$
\text{total number of points evaluated in batch } b
= \text{batch size} \times \text{num points per shape}
$$

Since each sample is a point cloud with a fixed number of points, this metric tells us how often PointNet assigns the correct part label at the point level.

### eval mIoU of {shape}

This is the mean IoU for one object category, such as Airplane or Chair.

For each valid part $p$ of that category:

$$
\text{IoU}_p = \frac{|\text{pred}_p \cap \text{groundtruth}_p|}{|\text{pred}_p \cup \text{groundtruth}_p|}
$$

Then for one object instance:

$$
\text{mIoU}(\text{instance}) = \frac{1}{P}\sum_{p=1}^{P} \text{IoU}_p
$$

where $P$ is the number of parts for that shape category.

Then for a shape category:

$$
\text{eval mIoU of shape} = \text{average of instance mIoUs for that category}
$$

So for example, **eval mIoU of Chair** means the average part-segmentation IoU over all chair objects in the evaluation set.

A common convention in this repo is that if a part is absent in both prediction and ground truth, its IoU is taken as $1.0$.

### Epoch X test Accuracy

This is the pointwise accuracy on the validation/test split at the end of that epoch.

$$
\text{Test Accuracy} = \frac{\text{total correctly predicted points}}{\text{total evaluated points}}
$$

It measures overall point classification correctness, but it does not capture part overlap quality as well as IoU does.

### Class avg mIoU

This averages the category-level mIoUs equally across all object categories.

$$
\text{Class Avg mIoU} = \frac{1}{C}\sum_{c=1}^{C} \text{mIoU}_c
$$

where $C$ is the number of object categories and $\text{mIoU}_c$ is the average instance mIoU for category $c$.

This metric treats each category equally, so categories with fewer samples still matter just as much as categories with many samples.

### Instance avg mIoU

This averages mIoU over all individual object instances, regardless of category.

$$
\text{Instance Avg mIoU} = \frac{1}{N}\sum_{i=1}^{N} \text{mIoU}_i
$$

where $N$ is the total number of object instances.

This reflects overall segmentation quality across the whole dataset, but categories with more samples have more influence.

### Best accuracy

The highest test accuracy achieved so far across epochs.

### Best class avg mIoU

The highest class average mIoU achieved so far across epochs.

### Best instance avg mIoU

The highest instance average mIoU achieved so far across epochs.

---

## Test / evaluation log

The evaluation script reports the same segmentation metrics, but now on the test set only.

### eval mIoU of {shape}

Same definition as above: average instance mIoU for that object category.

### Accuracy

Overall pointwise test accuracy:

$$
\text{Accuracy} = \frac{\text{total correctly predicted points}}{\text{total tested points}}
$$

### Class avg accuracy

Average of the per-class accuracies:

$$
\text{Class Avg Accuracy} = \frac{1}{K}\sum_{k=1}^{K} \text{Accuracy}_k
$$

where $K$ is the number of classes and $\text{Accuracy}_k$ is the accuracy for class $k$.

Unlike overall accuracy, this gives each class equal weight.

### Class avg mIoU

Same as above:

$$
\text{Class Avg mIoU} = \frac{1}{C}\sum_{c=1}^{C} \text{mIoU}_c
$$

### Instance avg mIoU

Same as above:

$$
\text{Instance Avg mIoU} = \frac{1}{N}\sum_{i=1}^{N} \text{mIoU}_i
$$

---

## Interpretation for PointNet

For PointNet, these metrics tell us slightly different things:

- **Accuracy** tells us how many individual points were labeled correctly.
- **IoU / mIoU** tells us how well the predicted parts overlap the true parts, which is usually more informative for segmentation.
- **Class avg mIoU** shows whether the model performs well across all object categories fairly.
- **Instance avg mIoU** shows the overall average segmentation quality across all test shapes.

## Evaluation summary

Because PointNet performs part segmentation, the main evaluation metrics are pointwise accuracy and mean IoU. Accuracy measures how many points are labeled correctly, while mIoU measures how well the predicted parts overlap the ground-truth parts, both per category and across all instances.

## Summary

In summary, the original repository was adapted to work with the local ShapeNet Part dataset and an Apple Silicon MPS environment. This setup provides a working PointNet baseline that can later be compared against PointNet++ and other future experiments.
