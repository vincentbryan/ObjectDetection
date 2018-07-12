import sys
import numpy as np
import pcl
import glob
import Parameter as param
import Transformer


def load_from_pcd(path):
    point_cloud = pcl.load(path)
    return np.array(list(point_cloud), dtype=np.float32)


def load_from_bin(path):
    point_cloud = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return point_cloud


# Extract ROI: the sector in front of the vehicle
def angle_filter(data):
    x_dim = np.logical_and(data[:, 0] > param.X_RANGE[0], data[:, 0] < param.X_RANGE[1])
    y_dim = np.logical_and(data[:, 1] > param.Y_RANGE[0], data[:, 1] < param.Y_RANGE[1])
    z_dim = np.logical_and(data[:, 2] > param.Z_RANGE[0], data[:, 2] < param.Z_RANGE[1])

    res_idx = np.logical_and(x_dim, np.logical_and(y_dim, z_dim))

    return data[res_idx]


def read_label_from_txt(file):

    boundary_boxes = []

    with open(file, 'r') as f:

        labels = f.read().split("\n")

        for label in labels:

            if not label:
                continue
            label = label.split(" ")

            if label[0] == "DontCare":
                continue
            if label[0] in param.TARGET_LIST:
                boundary_boxes.append(label[8:15])

    if boundary_boxes:
        data = np.array(boundary_boxes, dtype=np.float32)
        return data[:, 3:6], data[:, :3], data[:, 6]
    else:
        return None, None, None


def label_parser(label_file, rt_cam_to_vel):

    location = []
    shape = [] # height, width, length
    rotation = []

    if param.LABEL_FORMAT == 'txt':

        location, shape, rotation = read_label_from_txt(label_file)

        if len(location) == 0 or len(shape) == 0 or len(rotation) == 0:
            return None, None, None

        rotation = np.pi / 2 - rotation
        location = np.c_[location, np.ones(location.shape[0])]

        location = np.dot(rt_cam_to_vel, location.transpose())[:3, :]
        location = location.transpose()

        location_idx = Transformer.clipper(location, rotation, shape)

        if len(location_idx) == 0:
            return None, None, None

        location = location[location_idx]

    else:
        print "Invalid label format!!!"

    return location, rotation, shape


def calib_parser(calib_file):

    data = {}

    with open(calib_file, 'r') as f:
        for line in f.readlines():
            if not line or line == "\n":
                continue
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    rect = data["R0_rect"].reshape(3, 3)
    inverse_rect = np.linalg.inv(rect)

    vel_to_cam = data["Tr_velo_to_cam"].reshape(3, 4)
    cam_to_vel = np.ones_like(vel_to_cam)
    cam_to_vel[:, :3] = np.linalg.pinv(vel_to_cam[:, :3])
    cam_to_vel[:, 3] = -1 * vel_to_cam[:, 3]

    return np.dot(inverse_rect, cam_to_vel)


def get_next_batch(data_path, label_path, calib_path):

    data_path_ = glob.glob(data_path + "*")
    label_path_ = glob.glob(label_path + "*")
    calib_path_ = glob.glob(calib_path + "*")

    data_path_.sort()
    label_path_.sort()
    calib_path_.sort()

    iter_times = len(label_path_) // param.BATCH_SIZE

    for iter in range(iter_times):

        voxel_batch = []
        label_batch = []

        start_idx = iter * param.BATCH_SIZE
        end_idx = (iter+1) * param.BATCH_SIZE

        print "- - - - - - - - - - -"
        print ("  Batch : %2d/%2d    " % (iter, iter_times))
        print "- - - - - - - - - - -"
        obj_cnt = 0
        for iter_data_path, iter_label_path, iter_calib_path in zip(data_path_[start_idx:end_idx],
                                                                    label_path_[start_idx:end_idx],
                                                                    calib_path_[start_idx:end_idx]):

            point_cloud = None
            rt_cam_to_vel = None
            location = None
            rotation = None
            shape = None
            boundary_boxes = None



            if param.DATA_FORMAT == 'pcd':
                point_cloud = load_from_pcd(iter_data_path)
            elif param.DATA_FORMAT == 'bin':
                point_cloud = load_from_bin(iter_data_path)

            if iter_calib_path:
                rt_cam_to_vel = calib_parser(iter_calib_path)

            if iter_label_path:
                location, rotation, shape = label_parser(iter_label_path, rt_cam_to_vel)

            point_cloud = angle_filter(point_cloud)

            voxel = Transformer.raw_to_voxel(point_cloud)

            objectness_label = Transformer.get_objectness_label(location, rotation, shape)

            voxel_batch.append(voxel)
            label_batch.append(objectness_label)
            obj_cnt += location.shape[0]

            print("Loading : %s \tobj : %d" % (iter_data_path.split("/")[-1], location.shape[0]))

        yield voxel_batch, label_batch, obj_cnt


def get_test_voxel(data_path):

    point_cloud = None

    if param.DATA_FORMAT == 'pcd':
        point_cloud = load_from_pcd(data_path)
    elif param.DATA_FORMAT == 'bin':
        point_cloud = load_from_bin(data_path)

    point_cloud = angle_filter(point_cloud)

    voxel = Transformer.raw_to_voxel(point_cloud)

    return point_cloud, voxel[np.newaxis, :]


def get_visualize_input(data_path, calib_path, label_path):

    point_cloud = None
    rt_cam_to_vel = None
    position = None
    rotation = None
    shape = None

    if param.DATA_FORMAT == 'pcd':
        point_cloud = load_from_pcd(data_path)
    elif param.DATA_FORMAT == 'bin':
        point_cloud = load_from_bin(data_path)

    if calib_path:
        rt_cam_to_vel = calib_parser(calib_path)

    if label_path:
        position, rotation, shape = label_parser(label_path, rt_cam_to_vel)

    point_cloud = angle_filter(point_cloud)
    gt_objectness = Transformer.get_objectness_label(position, rotation, shape)
    return point_cloud, gt_objectness[np.newaxis, :]


