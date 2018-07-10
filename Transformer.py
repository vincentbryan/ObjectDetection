import sys
import numpy as np
import pcl
import Parameter as param
from sensor_msgs.msg import PointCloud2

def clipper(location, rotation, shape):
    x_clipper = np.logical_and(location[:, 0] >= param.X_RANGE[0], location[:, 0] < param.X_RANGE[1])
    y_clipper = np.logical_and(location[:, 1] >= param.Y_RANGE[0], location[:, 1] < param.Y_RANGE[1])
    z_clipper = np.logical_and(location[:, 2] + shape[:, 0]/2 >= param.Z_RANGE[0],
                           location[:, 2] + shape[:, 0]/2 < param.Z_RANGE[1])
    index_clipped = np.logical_and(x_clipper, y_clipper, z_clipper)
    return index_clipped


def raw_to_voxel(point_cloud):

    resolution = param.RESOLUTION

    x = param.X_RANGE
    y = param.Y_RANGE
    z = param.Z_RANGE

    dim_x = np.logical_and(point_cloud[:, 0] >= x[0], point_cloud[:, 0] < x[1])
    dim_y = np.logical_and(point_cloud[:, 1] >= y[0], point_cloud[:, 1] < y[1])
    dim_z = np.logical_and(point_cloud[:, 2] >= z[0], point_cloud[:, 2] < z[1])

    point_cloud = point_cloud[:, :3][np.logical_and(dim_x, np.logical_and(dim_y, dim_z))]

    point_cloud = ((point_cloud - np.array([x[0], y[0], z[0]])) / resolution).astype(np.int32)

    voxel = np.zeros((param.VOXEL_SHAPE[0], param.VOXEL_SHAPE[1], param.VOXEL_SHAPE[2], 1))
    voxel[point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 0] = 1

    return voxel


def label_to_voxel(location, shape, raw_boxes):

    x_dim = np.logical_and(location[:, 0] >= param.X_RANGE[0], location[:, 0] < param.X_RANGE[1])
    y_dim = np.logical_and(location[:, 1] >= param.Y_RANGE[0], location[:, 1] < param.Y_RANGE[1])
    z_dim = np.logical_and(location[:, 2] + shape[:, 0]/2 >= param.Z_RANGE[0],
                           location[:, 2] + shape[:, 0]/2 < param.Z_RANGE[1])
    voxel_filtered = np.logical_and(x_dim, y_dim, z_dim)

    location2 = location.copy()
    location2[:, 2] += shape[:, 0]/2

    training_boxes = raw_boxes[voxel_filtered].copy()
    location_uniformed = (location2 - [param.X_RANGE[0], param.Y_RANGE[0], param.Z_RANGE[0]]).astype(np.int)
    anchor_center = location_uniformed + [param.X_RANGE[0], param.Y_RANGE[0], param.Z_RANGE[0]]

    for idx, (p1, p2) in enumerate( zip(raw_boxes[voxel_filtered], anchor_center) ):
        training_boxes[idx] = p1 - p2

    return location_uniformed, training_boxes


def get_objectness_label(location, rotation, shape):

    # boxes = []
    #
    # for iter_loc, iter_rot, iter_shape in zip(location, rotation, shape):
    #
    #     x, y, z = iter_loc
    #     h, w, l = iter_shape
    #     if l > 10:
    #         continue
    #
    #     # 8 corners
    #     box = np.array([
    #         [x - l / 2., y - w / 2., z],
    #         [x + l / 2., y - w / 2., z],
    #         [x - l / 2., y + w / 2., z],
    #         [x - l / 2., y - w / 2., z + h],
    #         [x - l / 2., y + w / 2., z + h],
    #         [x + l / 2., y + w / 2., z],
    #         [x + l / 2., y - w / 2., z + h],
    #         [x + l / 2., y + w / 2., z + h],
    #     ])
    #
    #     # rotate with y_rotation
    #     rotate_matrix = np.array([
    #         [np.cos(iter_rot), -np.sin(iter_rot), 0],
    #         [np.sin(iter_rot), np.cos(iter_rot), 0],
    #         [0, 0, 1]
    #     ])
    #
    #     res = np.dot(rotate_matrix, box.transpose())
    #
    #     boxes.append(res.transpose())

    # objectness_label = (location[index_clipped] - [param.X_RANGE[0], param.Y_RANGE[0], param.Z_RANGE[0]]).astype(np.int)

    # Initial the object map as background
    objectness_label = np.zeros((param.ANCHOR_SHAPE[0], param.ANCHOR_SHAPE[1], param.ANCHOR_SHAPE[2], 2))
    objectness_label[:, :, :, 1] = 1

    idx = vel_coordinate_to_anchor(location)
    objectness_label[idx[:, 0], idx[:, 1], idx[:, 2], 0] = 1
    objectness_label[idx[:, 0], idx[:, 1], idx[:, 2], 1] = 0

    return objectness_label


def vel_coordinate_to_anchor(position):
    # TODO
    scale = param.STRIDE_STEP ** 4
    anchor_idx = ((position - [param.X_RANGE[0], param.Y_RANGE[0], param.Z_RANGE[0]]) / scale).astype(np.int32)
    return anchor_idx


def anchor_to_vel_coordinate(index):
    position = (index + [param.X_RANGE[0], param.Y_RANGE[0], param.Z_RANGE[0]]) * param.STRIDE_STEP ** 4
    return position