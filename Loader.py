import sys
import numpy as np
import pcl
import glob
import Parameter as param
import Transformer
import rospy
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


def load_from_pcd(path):
    point_cloud = pcl.load(path)
    return np.array(list(point_cloud), dtype=np.float32)


def load_from_bin(path):
    point_cloud = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return point_cloud


# Extract ROI: the sector in front of the vehicle
def angle_filter(data):
    res = np.logical_and( (data[:, 1] < data[:, 0] - 0.27),
                          (-data[:, 1] < data[:, 0] - 0.27))
    return data[res]


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
        # TODO Optimizer
        if len(location) == 0:
            return None, None, None
        elif len(shape) == 0:
            return None, None, None
        elif len(rotation) == 0:
            return None, None, None

        rotation = np.pi / 2 - rotation
        location = np.c_[location, np.ones(location.shape[0])]

        location = np.dot(rt_cam_to_vel, location.transpose())[:3, :]
        location = location.transpose()
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

    data_path_ = glob.glob(data_path)
    label_path_ = glob.glob(label_path)
    calib_path_ = glob.glob(calib_path)

    data_path_.sort()
    label_path_.sort()
    calib_path_.sort()

    iter_times = len(label_path_) // param.BATCH_SIZE

    for iter in range(iter_times):

        voxel_batch = []
        label_batch = []

        start_idx = iter * param.BATCH_SIZE
        end_idx = (iter+1) * param.BATCH_SIZE

        print ("Dealing Batch : %d " % iter)

        for iter_data_path, iter_label_path, iter_calib_path in zip(data_path_[start_idx:end_idx],
                                                                    label_path_[start_idx:end_idx],
                                                                    calib_path_[start_idx:end_idx]):



            point_cloud = None
            rt_cam_to_vel = None
            location = None
            rotation = None
            shape = None
            boundary_boxes = None

            print "Dealing data : " + iter_data_path.split("/")[-1]

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

        yield voxel_batch, label_batch

        # yield np.array(batch_voxel, dtype=np.float32)[:, :, :, :, np.newaxis], np.array(batch_obj_map, dtype=np.float32)


def publisher(point_cloud):

    pub = rospy.Publisher("/points_raw", PointCloud2, queue_size=100000)
    rospy.init_node("cnn_3d_point_cloud")
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "cnn_3d_point_cloud"
    points = pc2.create_cloud_xyz32(header, point_cloud[:, :3])

    sleep_rate = rospy.Rate(0.1)
    while not rospy.is_shutdown():
        pub.publish(points)
        sleep_rate.sleep()


