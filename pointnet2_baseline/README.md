For the PointNet++ baseline we used the following repo which is a PyTorch implementation of PointNet++: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

Environment used for training/evaluation was Google Colab. 

## Steps Followed 
- Download the ShapeNet dataset from the following link:
https://drive.usercontent.google.com/download?id=1W3SEE-dY1sxvlECcOwWSDYemwHEUbJIS&authuser=0
- Extract the dataset and upload to Google Drive. Ensure the folder is named shapenetcore_partanno_segmentation_benchmark_v0_normal and is located in Google Drive root (MyDrive).
- Create a new notebook in Google Colab
- Set runtime: Runtime -> Change runtime type -> T4 GPU
- Upload the file train_pointnet2_baseline.sh provided in this folder to the notebook environment.
- Run the following cells in the notebook
    - Mount Drive:

    `from google.colab import drive`

    `drive.mount('/content/drive')`

    - Run the training script

    `!bash train_pointnet2_baseline.sh`

- Training log in the notebook environment can be found at: Pointnet_Pointnet2_pytorch/log/part_seg/pointnet2_part_seg_msg_xyz_1024_bs20/logs/pointnet2_part_seg_msg.txt

The training log file has been downloaded and saved as train_log.txt in this folder.

- Evaluation log in the notebook environment can be found at: Pointnet_Pointnet2_pytorch/log/part_seg/pointnet2_part_seg_msg_xyz_1024_bs20/eval.txt

The evaluation log file has been downloaded and saved as eval_log.txt in this folder.


## Explanation of metrics in logs

## Training Log:

The following metrics are reported for *each epoch* in training:
### "Train accuracy":
$$
    \text{Train Accuracy} = \frac{1}{B} \sum_{b=1}^{B} \frac{\text{num of correct predictions in batch b}}{\text{batch size * num points}}
$$
Note: num_points is a constant value describing the number of points used to form the point cloud of a shape sample. The batch contains batch_size number of point clouds. 
Also, the predictions are made for each point to classify their part. 

### "eval mIoU of {shape}":
For each instance (ie. one object): 

For each part p belonging to that shape category, take the Intersection over Union (IoU):
$$
    IoU_p = \frac{|pred_p ∩ groundtruth_p|}{|pred_p ∪ groundtruth_p|}
$$
Then: 
$$
    mIoU(instance) = \frac{1}{P}\sum_{p=1}^{P}IoU_p
$$
Where P is the number of parts for that shape category.

Then for a shape category (eg. Airplane):
$$
    \text{eval mIoU of shape = mean of instance mIoUs for that category}
$$

Note: In case a part p doesn't exist in both ground truth and prediction, IoU_p = 1.0.

### "epoch {epoch_num} test Accuracy":
$$
    \text{Test Accuracy} = \frac{\text{Total correct predictions}}{\text{Total num of points across all batches}}
$$
Where total num of points is the sum of (batch_size*num_points) across all batches.

### "Class avg mIOU":
$$
    \text{Class Avg mIoU} = \frac{1}{C}\sum_{c=1}^{C}mIoU_c
$$
Where c iterates over all shape categories (Airplane, Chair, etc.) and mIoU_c is the average over instances of class c (as were reported above).

### "Instance avg mIOU":
$$
    \text{Instance Avg mIoU} = \frac{1}{N}\sum_{i=1}^{N}mIoU_i
$$
Note: This average is across instances belonging to all shape categories, without considering class groupings. 

### "Best accuracy":
- Highest test accuracy across all epochs so far.

### "Best class avg mIOU":
- Highest class avg mIOU across all epochs so far.

### "Best instance avg mIOU":
- Highest instance avg mIOU across all epochs so far.


## Evaluation Log:

The following metrics are reported for evaluation:
### "eval mIoU of {shape}":
- Same definition as described in training log above.

### "Accuracy":
- Test accuracy. Same definition as described in training log above.

### "Class avg accuracy":
$$
    \text{Class Avg Accuracy} = \frac{1}{K} \sum_{k=1}^{K} \frac{\text{num of correct predictions in class k}}{\text{total num of points in class k}}
$$
Where K is the total number of part classes from all shape categories. In this dataset, K=50.

### "Class avg mIOU":
- Same definition as described in training log above.

### "Instance avg mIOU":
- Same definition as described in training log above.