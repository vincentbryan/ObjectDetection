import tensorflow as tf
import Models
import Loader
import Parameter as param
from tensorflow.python import debug as tf_debug
import rospy
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2


def create_optimizer(loss, learning_rate=0.001):
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return opt


def Train():
    with tf.Session() as sess:

        voxel_input = tf.placeholder(tf.float32, [None, param.VOXEL_SHAPE[0], param.VOXEL_SHAPE[1], param.VOXEL_SHAPE[2], 1])
        label_input = tf.placeholder(tf.float32, [None, param.ANCHOR_SHAPE[0], param.ANCHOR_SHAPE[1], param.ANCHOR_SHAPE[2], 2], name="label_input")

        cnn_3d = Models.cnn_3d(sess, voxel_input)

        sum = tf.multiply(label_input, tf.log(cnn_3d.objectness_predict), name="sum")
        obj_loss = -tf.reduce_sum(sum, name="obj_loss")
        optimizer = create_optimizer(obj_loss)

        init = tf.global_variables_initializer()
        sess.run(init)

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        for epoch in range(param.EPOCH):
            for (voxel_batch, label_batch) in Loader.get_next_batch(param.DATA_DIR, param.LABEL_DIR, param.CALIB_DIR):
                _, loss_value = sess.run([optimizer, obj_loss], feed_dict={voxel_input: voxel_batch, label_input:label_batch})
                print("Epoch : {}, obj_loss : {}".format(epoch, loss_value))


if __name__ == '__main__':

    Train()

    """
    pc_pub = rospy.Publisher("/points_raw", PointCloud2, queue_size=100000)
    rospy.init_node("cnn_3d_point_cloud")
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "cnn_3d"

    marker_pub = rospy.Publisher("anchor_obj", Marker, queue_size=10)

    sleep_rate = rospy.Rate(1)

    while not rospy.is_shutdown():

        point_cloud, obj_position = Loader.get_next_batch(param.DATA_DIR, param.LABEL_DIR, param.CALIB_DIR).next()

        points = pc2.create_cloud_xyz32(header, point_cloud[:, :3])

        marker = Marker()
        marker.header.frame_id = "cnn_3d"
        marker.header.stamp = rospy.Time.now()

        marker.ns = "basic_shapes"
        marker.id = 0
        marker.type = Marker.CUBE

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.4

        marker.action = Marker.ADD

        marker.pose.position.x = obj_position[0, 0]
        marker.pose.position.y = obj_position[0, 1]
        marker.pose.position.z = obj_position[0, 2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.color.r = 0.0
        marker.color.g = 0.8
        marker.color.b = 0.0
        marker.color.a = 1.0

        pc_pub.publish(points)
        marker_pub.publish(marker)

        sleep_rate.sleep()
    """










