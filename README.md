# 3d-point-cloud-segmentation
Deep learning project implementing PointNet++ from scratch for fine-grained 3D point cloud segmentation and comparing against existing PointNet and PointNet++ baselines.

## Setup
Clone this repo to your preferred location.

### Installation
Install the requirements listed in requirements.txt using the following:

`pip install -r requirements.txt`

### Data 
Download the official ShapeNetPart dataset from the following link 

https://drive.usercontent.google.com/download?id=1W3SEE-dY1sxvlECcOwWSDYemwHEUbJIS&authuser=0

and save in `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`.

## Usage
### PointNet++ Classification Task
#### Training
For training, run the following command `python train_pointnetpp_classification.py` with the following optional arguments:
- `--data-root` (Path to dataset)
- `--num-points` (Number of points in each point cloud)
- `--use-normals` (Flag; if set, uses surface normals as additional input features ie. 6 input channels instead of 3)
- `--batch-size` (Batch size)
- `--epochs` (Number of epochs to train for)
- `--lr` (Learning rate)
- `--weight-decay` (Weight decay)
- `--num-workers` (Number of DataLoader workers)
- `--save-dir` (Directory to save model checkpoints)
- `--sa-aggregation` (Set abstraction local features grouping method: `mrg` (multiresolution) or `msg` (multiscale))

#### Testing
For testing a trained model, run the following command `python test_pointnetpp_classification.py` with the following arguments:
- `--checkpoint` (Required. Path to trained model checkpoint)
- `--data-root` (Path to dataset)
- `--num-points` (Number of points in each point cloud)
- `--use-normals` (Flag; if set, uses surface normals as additional input features ie. 6 input channels instead of 3)
- `--batch-size` (Batch size)
- `--num-workers` (Number of DataLoader workers)

### PointNet++ Part Segmentation Task
#### Training
For training, run the following command `python train_pointnetpp_part_segmentation.py` with the following optional arguments:
- `--data-root` (Path to dataset)
- `--num-points` (Number of points in each point cloud)
- `--use-normals` (Flag; if set, uses surface normals as additional input features ie. 6 input channels instead of 3)
- `--batch-size` (Batch size)
- `--epochs` (Number of epochs to train for)
- `--lr` (Learning rate)
- `--weight-decay` (Weight decay)
- `--num-workers` (Number of DataLoader workers)
- `--num-part-classes` (Total number of part classes, eg. 50 in ShapeNet Part dataset)
- `--use-category-conditioning` (Flag; if set, conditions the model on the object category label)
- `--category-embed-dim` (Embedding dimension for the category label - only used with `--use-category-conditioning`)
- `--save-dir` (Directory to save model checkpoints)
- `--sa-aggregation` (Set abstraction local features grouping method: `mrg` (multiresolution) or `msg` (multiscale))

#### Testing
For testing a trained model, run the following command `python test_pointnetpp_part_segmentation.py` with the following arguments:
- `--checkpoint` (Required. Path to trained model checkpoint)
- `--data-root` (Path to dataset)
- `--num-points` (Number of points in each point cloud)
- `--use-normals` (Flag; if set, uses surface normals as additional input features ie. 6 input channels instead of 3)
- `--batch-size` (Batch size)
- `--num-workers` (Number of DataLoader workers)
- `--sa-aggregation` (Set abstraction local features grouping method: `mrg` (multiresolution) or `msg` (multiscale))

### Outputs
Default locations of outputs are as follows:
- Train and test logs can be found in `/logs`
- Model checkpoints can be found in `/checkpoints`
- Training graphs can be found in `/graphs`
- 3d point cloud plots can be found in `/visuals`
    - Visuals are generated after part segmentation inference, and each visual shows a predicted vs ground truth side-by-side comparison of a point cloud, with points colour-coded by part label.

## Baselines
For training and evaluation of the baseline PointNet and PointNet++ architectures, the following repo of a PyTorch implementation was used:
    https://github.com/yanx27/Pointnet_Pointnet2_pytorch

Detailed documentation and steps followed for baseline PointNet and baseline PointNet++ can be found in the READMEs in `/pointnet_baseline` and `/pointnet2_baseline` respectively. 

## Key References
- Original PointNet architecture: https://github.com/charlesq34/pointnet (repo), https://arxiv.org/pdf/1612.00593 (paper)
- Original PointNet++ architecture: https://github.com/charlesq34/pointnet2# (repo), https://arxiv.org/pdf/1706.02413 (paper)
- PyTorch Implementation used for baselines: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- Official ShapeNet Part dataset: https://drive.usercontent.google.com/download?id=1W3SEE-dY1sxvlECcOwWSDYemwHEUbJIS&authuser=0