#!/bin/bash
set -euo pipefail

# -------- CONFIG --------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR=${1:-pointnet-baseline-repo}
DATASET_PATH=${2:-"$SCRIPT_DIR/../data/shapenetcore_partanno_segmentation_benchmark_v0_normal"}
TRAIN_LOG_DIR=${3:-pointnet_partseg_xyz_1024_bs20_train}

# -------- SETUP --------
if [ ! -d "$DATASET_PATH" ]; then
    echo "Dataset path does not exist: $DATASET_PATH"
    exit 1
fi

if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git "$REPO_DIR"
fi

if [ ! -f "$DATASET_PATH/synsetoffset2category.txt" ]; then
    echo "Dataset folder is missing synsetoffset2category.txt: $DATASET_PATH"
    exit 1
fi

cd "$REPO_DIR"

# Install dependencies
pip install torch torchvision torchaudio numpy h5py matplotlib tqdm

# -------- PATCH DATASET PATHS --------
sed -i '' "s#root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'#root = '$DATASET_PATH/'#" train_partseg.py
sed -i '' "s#root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'#root = '$DATASET_PATH/'#" test_partseg.py
sed -i '' "s#root = '\./data/shapenetcore_partanno_segmentation_benchmark_v0_normal'#root = '$DATASET_PATH'#" data_utils/ShapeNetDataLoader.py
# -------- PATCH NUMPY DEPRECATION --------
sed -i '' 's/np.float/float/g' train_partseg.py
sed -i '' 's/np.float/float/g' test_partseg.py

# -------- PATCH DEVICE SELECTION IN train_partseg.py --------
python - <<'PY'
from pathlib import Path
p = Path("train_partseg.py")
s = p.read_text()

if 'torch.backends.mps.is_available()' not in s:
    s = s.replace(
        "def main(args):",
        '''def main(args):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)
'''
    )

s = s.replace(
    "classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()",
    "classifier = MODEL.get_model(num_part, normal_channel=args.normal).to(device)"
)

s = s.replace(
    "classifier = classifier.cuda()",
    "classifier = classifier.to(device)"
)

s = s.replace(
    "criterion = MODEL.get_loss().cuda()",
    "criterion = MODEL.get_loss().to(device)"
)

s = s.replace(
    "points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()",
    "points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)"
)

s = s.replace(
    "points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()",
    "points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)"
)

s = s.replace(
    "seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes).cuda())",
    "seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes).to(device))"
)

if "def to_categorical(y, num_classes):\n    return torch.eye(num_classes, device=y.device)[y, :]" not in s:
    import re
    s = re.sub(
        r"def to_categorical\(.*?\n(?:    .*?\n)+",
        'def to_categorical(y, num_classes):\n    return torch.eye(num_classes, device=y.device)[y, :]\n\n',
        s,
        count=1
    )

p.write_text(s)
PY

# -------- PATCH DEVICE SELECTION IN test_partseg.py --------
python - <<'PY'
from pathlib import Path
p = Path("test_partseg.py")
s = p.read_text()

if 'torch.backends.mps.is_available()' not in s:
    s = s.replace(
        "def main(args):",
        '''def main(args):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)
'''
    )

s = s.replace(
    "classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()",
    "classifier = MODEL.get_model(num_part, normal_channel=args.normal).to(device)"
)

s = s.replace(
    "classifier = classifier.cuda()",
    "classifier = classifier.to(device)"
)

s = s.replace(
    "checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')",
    """checkpoint = torch.load(
        str(experiment_dir) + '/checkpoints/best_model.pth',
        map_location=device,
        weights_only=False
    )"""
)

s = s.replace(
    "points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()",
    "points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)"
)

s = s.replace(
    "seg_pred, _ = classifier(points, to_categorical(label, num_classes).cuda())",
    "seg_pred, _ = classifier(points, to_categorical(label, num_classes).to(device))"
)

s = s.replace(
    "vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()",
    "vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part, device=device)"
)

if "def to_categorical(y, num_classes):\n    return torch.eye(num_classes, device=y.device)[y, :]" not in s:
    import re
    s = re.sub(
        r"def to_categorical\(.*?\n(?:    .*?\n)+",
        'def to_categorical(y, num_classes):\n    return torch.eye(num_classes, device=y.device)[y, :]\n\n',
        s,
        count=1
    )

p.write_text(s)
PY

# -------- PATCH pointnet_utils.py --------
python - <<'PY'
from pathlib import Path
p = Path("models/pointnet_utils.py")
s = p.read_text()

s = s.replace(
    """iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden""",
    """iden = torch.eye(3, dtype=x.dtype, device=x.device).view(1, 9).repeat(batchsize, 1)
        x = x + iden"""
)

s = s.replace(
    """iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden""",
    """iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k * self.k).repeat(batchsize, 1)
        x = x + iden"""
)

s = s.replace(
    """def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss""",
    """def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d, dtype=trans.dtype, device=trans.device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss"""
)

p.write_text(s)
PY

# -------- TRAIN --------
python train_partseg.py \
    --model pointnet_part_seg \
    --npoint 1024 \
    --batch_size 20 \
    --epoch 5 \
    --log_dir "$TRAIN_LOG_DIR" | tee pointnet_part_seg.txt

# -------- EVALUATE --------
python test_partseg.py \
    --log_dir "$TRAIN_LOG_DIR" \
    --num_point 1024 \
    --batch_size 20 | tee eval.txt

echo "Done."
echo "Training log saved to: pointnet_part_seg.txt"
echo "Evaluation log saved to: eval.txt"