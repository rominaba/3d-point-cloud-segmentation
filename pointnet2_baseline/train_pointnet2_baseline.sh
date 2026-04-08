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

# Copy visualize.py into the cloned repo
cp /content/drive/MyDrive/visualize.py .

# Modify test_partseg.py to add in visualization code
python3 << EOF
import io

file_path = "test_partseg.py"

with open(file_path, "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)

    # 1. Add import
    if "from visualize import visualize_part_seg_comparison" not in "".join(lines):
        if "import numpy as np" in line:
            new_lines.append("from visualize import visualize_part_seg_comparison\n")

    # 2. Add saved_categories_for_visualize
    if "saved_categories_for_visualize" not in "".join(lines):
        if "classifier = classifier.eval()" in line:
            new_lines.append("        saved_categories_for_visualize = set()\n")

    # 3. Add visualization block
    if "visualize_part_seg_comparison(points_np, pred_np" not in "".join(lines):
        if "shape_ious[cat].append(np.mean(part_ious))" in line:
            new_lines.append("""                
                cat_name = seg_label_to_cat[target[i, 0]]
                if cat_name not in saved_categories_for_visualize:
                    points_np = points[i].cpu().numpy().transpose(1, 0)
                    pred_np = cur_pred_val[i]
                    gt_np = target[i]
                    visualize_part_seg_comparison(points_np, pred_np, gt_np, title=cat_name, save_path=f"log/part_seg/{args.log_dir}/visuals/{cat_name}_example.png")
                    saved_categories_for_visualize.add(cat_name)
""")

with open(file_path, "w") as f:
    f.writelines(new_lines)
EOF

# Create a folder to store the plots
mkdir -p log/part_seg/$LOG_DIR/visuals

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