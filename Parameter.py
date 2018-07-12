STRIDE_STEP = 2

# A voxel cube is 0.2 meter long
RESOLUTION = 0.2

# 4 is the layer depth
SCALE = STRIDE_STEP ** 4

X_RANGE = (0, 16)
Y_RANGE = (-8, 8)
Z_RANGE = (-1.3, 1.9)

VOXEL_SHAPE_X = int(round((X_RANGE[1] - X_RANGE[0]) / RESOLUTION))
VOXEL_SHAPE_Y = int(round((Y_RANGE[1] - Y_RANGE[0]) / RESOLUTION))
VOXEL_SHAPE_Z = int(round((Z_RANGE[1] - Z_RANGE[0]) / RESOLUTION))

VOXEL_SHAPE = (VOXEL_SHAPE_X, VOXEL_SHAPE_Y, VOXEL_SHAPE_Z)

ANCHOR_SHAPE = (VOXEL_SHAPE[0]/SCALE, VOXEL_SHAPE[1]/SCALE, VOXEL_SHAPE[2]/SCALE)

BATCH_SIZE = 10

DATA_DIR = "/media/vincent/DATA/Ubuntu/Project/Dataset/KITTI/VelodynePCD/debug/"
CALIB_DIR = "/media/vincent/DATA/Ubuntu/Project/Dataset/KITTI/VelodyneCalib/debug/"
LABEL_DIR = "/media/vincent/DATA/Ubuntu/Project/Dataset/KITTI/VelodyneLabel/debug/"

MODEL_DIR = "model/cnn_3d.ckpt"

DATA_FORMAT = "pcd"
LABEL_FORMAT = "txt"

TARGET_LIST = ["Car", "Van", "Pedestrian", "Truck", "Person_sitting", "Cyclist", "Tram", "Misc"]

BASE_LEARNING_RATE = 0.1
DECAY_RATE = 0.90