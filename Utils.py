import rospy
import numpy as np
import std_msgs.msg
import Transformer
import Parameter as param

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


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


def get_marker_array(data_input, marker_array=MarkerArray(), type="anchor"):

    obj_pos_in_vel = None
    offset = None
    marker_size = None
    color = None

    if type == "anchor":
        obj_pos_in_anchor = np.argwhere(data_input[:, :, :, :, 0] > data_input[:, :, :, :, 1])[:, 1:4]
        obj_pos_in_vel = Transformer.anchor_to_vel_coordinate(obj_pos_in_anchor)
        offset = param.STRIDE_STEP / 2.0
        marker_size = param.RESOLUTION * param.SCALE
        color = (1, 0, 0, 0.8)
    elif type == "voxel":
        obj_pos_in_voxel= np.argwhere(data_input[:, :, :, :, 0] > 0)[:, 1:4]
        obj_pos_in_vel = Transformer.voxel_to_vel_coordinate(obj_pos_in_voxel)
        offset = param.RESOLUTION / 2.0
        marker_size = param.RESOLUTION
        color = (0, 1, 0, 0.2)
    elif type == "ground_truth":
        obj_pos_in_vel = data_input
        offset = param.STRIDE_STEP / 2.0
        marker_size = param.RESOLUTION * param.SCALE
        color = (1, 0, 0, 0.8)

    marker_array_size = obj_pos_in_vel.shape[0]
    print ("Total %d objects detected" % marker_array_size)

    id = 0
    for i in range(marker_array_size):
        marker = Marker()
        marker.header.frame_id = "/cnn_3d"
        marker.header.stamp = rospy.Time.now()
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.id = id
        id += 1


        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        marker.scale.x = marker_size
        marker.scale.y = marker_size
        marker.scale.z = marker_size

        marker.pose.position.x = obj_pos_in_vel[i, 0] + offset
        marker.pose.position.y = obj_pos_in_vel[i, 1] + offset
        marker.pose.position.z = obj_pos_in_vel[i, 2] + offset

        marker_array.markers.append(marker)

    return marker_array


