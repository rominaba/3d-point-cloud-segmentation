For the PointNet++ baseline we used the following repo which is a PyTorch implementation of PointNet++: https://github.com/yanx27/Pointnet_Pointnet2_pytorch

Training was performed in Google Colab. 

Steps followed are as follows
- Download the ShapeNet dataset from the following link, extract and upload to Google Drive
https://drive.usercontent.google.com/download?id=1W3SEE-dY1sxvlECcOwWSDYemwHEUbJIS&authuser=0
- Create a new notebook in Google Colab
- Set Runtime -> Change runtime type -> T4 GPU
- Run the following cells in the notebook
    - Clone the repo of PyTorch PointNet++:
    
    `!git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch`

    `%cd Pointnet_Pointnet2_pytorch`

    - Install dependencies:

    `!pip install torch torchvision`

    `!pip install h5py matplotlib tqdm`

    `!pip install numpy`

    - Replace all np.float to float in train_partseg.py since np.float is deprecated in recent NumPy:

    `!sed -i 's/np.float/float/g' train_partseg.py`

    - Create data directory:

    `!mkdir -p data`

    - Mount Drive:

    `from google.colab import drive`

    `drive.mount('/content/drive')`

    - Link dataset from Drive to data folder:

    `!ln -s /content/drive/MyDrive/shapenetcore_partanno_segmentation_benchmark_v0_normal \
      data`

    - Train segmentation using the model for part segmentation with multi-scale grouping. Number of points set to 1024, batch size 20 and run for 5 epochs:

    `!python train_partseg.py \
    --model pointnet2_part_seg_msg \
    --npoint 1024 \
    --batch_size 20 \
    --epoch 5 \
    --log_dir pointnet2_part_seg_msg_xyz_1024_bs20_train`

Training results are copied to the train_log.txt file in this folder.
