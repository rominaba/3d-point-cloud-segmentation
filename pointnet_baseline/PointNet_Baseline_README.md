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

## Summary

In summary, the original repository was adapted to work with the local ShapeNet Part dataset and an Apple Silicon MPS environment. This setup provides a working PointNet baseline that can later be compared against PointNet++ and other future experiments.
