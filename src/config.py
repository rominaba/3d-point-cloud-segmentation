import os

DATA_ROOT = os.path.expanduser(
    "./data/shapenetcore_partanno_segmentation_benchmark_v0_normal"
)

NUM_POINTS = 1024
BATCH_SIZE = 16
USE_NORMALS = False
RANDOM_SEED = 42