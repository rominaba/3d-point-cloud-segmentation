For the PointNet++ baseline we used the following repo which is a PyTorch implementation of PointNet++: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

Training was performed in Google Colab. 

Steps followed are as follows
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

- Training log in the notebook environment can be found at: Pointnet_Pointnet2_pytorch/log/part_seg/pointnet2_part_seg_msg_xyz_1024_bs20_train/logs/pointnet2_part_seg_msg.txt

The log file has been downloaded and saved as train_log.txt in this folder.
