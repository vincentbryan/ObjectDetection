STRIDE_STEP = 2
X_RANGE = (0, 10)
Y_RANGE = (-5, 5)
Z_RANGE = (-3.0, 1.8)
VOXEL_SHAPE = (160, 160, 48)
ANCHOR_SHAPE = (VOXEL_SHAPE[0]/(STRIDE_STEP**4), VOXEL_SHAPE[1]/(STRIDE_STEP**4), VOXEL_SHAPE[2]/(STRIDE_STEP**4))

EPOCH = 101
BATCH_SIZE = 5

RESOLUTION = 0.2
SCALE = 4
LEARNING_RATE = 0.01

DATA_DIR = "/media/vincent/DATA/Ubuntu/Project/Dataset/KITTI/VelodynePCD/debug/*"
CALIB_DIR = "/media/vincent/DATA/Ubuntu/Project/Dataset/KITTI/VelodyneCalib/debug/*"
LABEL_DIR = "/media/vincent/DATA/Ubuntu/Project/Dataset/KITTI/VelodyneLabel/debug/*"

DATA_FORMAT = "pcd"
LABEL_FORMAT = "txt"

TARGET_LIST = ["Car", "Van", "Pedestrian"]