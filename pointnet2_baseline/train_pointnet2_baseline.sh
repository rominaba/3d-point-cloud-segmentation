#!/bin/bash
set -e

# ------CONFIG------
DATASET_PATH=${1:-/content/drive/MyDrive/shapenetcore_partanno_segmentation_benchmark_v0_normal}
LOG_DIR=pointnet2_part_seg_msg_xyz_1024_bs20

# ------SETUP------
# Clone repo
if [ ! -d "Pointnet_Pointnet2_pytorch" ]; then
    git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch
fi

cd Pointnet_Pointnet2_pytorch

# Install dependencies
pip install torch torchvision h5py matplotlib tqdm numpy

# Replace all np.float to float in train_partseg.py and test_partseg.py since np.float is deprecated in recent NumPy
sed -i 's/np.float/float/g' train_partseg.py
sed -i 's/np.float/float/g' test_partseg.py

# Add 'weights_only=False' to torch.load(...) in test_partseg.py in order to work with newer PyTorch versions
sed -i "s|torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')|torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', weights_only=False)|g" test_partseg.py

# Create data directory
mkdir -p data

# Link dataset from Drive to data folder
ln -sf $DATASET_PATH data

# ------TRAIN------
# Train segmentation using the model for part segmentation with multi-scale grouping. Number of points set to 1024, batch size 20 and run for 5 epochs
python train_partseg.py \
    --model pointnet2_part_seg_msg \
    --npoint 1024 \
    --batch_size 20 \
    --epoch 5 \
    --log_dir $LOG_DIR

# ------EVALUATE------
# Evaluate trained model with number of points set to 1024 and batch size 20.
python test_partseg.py \
    --num_point 1024 \
    --batch_size 20 \
    --log_dir $LOG_DIR