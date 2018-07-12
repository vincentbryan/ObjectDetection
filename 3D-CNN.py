import tensorflow as tf
import numpy as np
import Models
import Loader
import Utils
import Parameter as param
from tensorflow.python import debug as tf_debug
import rospy
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import PointCloud2


def create_optimizer(loss, cnt):

    learning_rate = tf.train.exponential_decay(
                        param.BASE_LEARNING_RATE,  # Base learning rate.
                        cnt * param.BATCH_SIZE,    # Current index into the dataset.
                        param.BATCH_SIZE,          # Decay step.
                        param.DECAY_RATE,          # Decay rate.
                        staircase=True, name="learning_rate")
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return opt


def Train():

    with tf.Session() as sess:

        voxel_input = tf.placeholder(tf.float32,
                                     [None, param.VOXEL_SHAPE[0], param.VOXEL_SHAPE[1], param.VOXEL_SHAPE[2], 1],
                                     name="voxel_input")
        label_input = tf.placeholder(tf.float32,
                                     [None, param.ANCHOR_SHAPE[0], param.ANCHOR_SHAPE[1], param.ANCHOR_SHAPE[2], 2],
                                     name="label_input")
        lr_step_input = tf.placeholder(tf.int32)

        cnn_3d = Models.cnn_3d(voxel_input)

        sum = tf.multiply(label_input, tf.log(cnn_3d.objectness_predict), name="sum")
        obj_loss = -tf.reduce_sum(sum, name="obj_loss")

        learning_rate = tf.train.exponential_decay(
            param.BASE_LEARNING_RATE,
            lr_step_input * param.BATCH_SIZE,
            param.BATCH_SIZE,
            param.DECAY_RATE,
            staircase=True, name="learning_rate")
        learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(obj_loss)

        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        writer = tf.summary.FileWriter("./log/", sess.graph)

        epoch_cnt = 0
        for (voxel_batch, label_batch, obj_num) in Loader.get_next_batch(param.DATA_DIR, param.LABEL_DIR, param.CALIB_DIR):

            epoch_cnt += 1

            _, loss_value, learning_rate_value = sess.run([optimizer, obj_loss, learning_rate],
                                                          feed_dict={voxel_input: voxel_batch,
                                                                     label_input:label_batch,
                                                                     lr_step_input:epoch_cnt})

            save_path = saver.save(sess, param.MODEL_DIR)

            print("Summary :")
            print("\ttot_obj_loss : {}".format(loss_value))
            print("\ttot_obj_numb : {}".format(obj_num))
            print("\tave_obj_loss : {}".format(loss_value / obj_num))
            print("\t          lr : {}".format(learning_rate_value))
            print("Model saved in path: %s\n" % save_path)

        writer.close()


def Test(voxel_path):
    rospy.init_node("cnn_3d")
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "cnn_3d"

    pc_pub = rospy.Publisher("cnn_3d_points_raw", PointCloud2, queue_size=100000)
    obj_array_pub = rospy.Publisher("cnn_3d_anchor_obj", MarkerArray, queue_size=1000)
    voxel_array_pub = rospy.Publisher("cnn_3d_voxel", MarkerArray, queue_size=1000)

    sleep_rate = rospy.Rate(1)

    point_cloud, voxel = Loader.get_test_voxel(voxel_path)
    points = pc2.create_cloud_xyz32(header, point_cloud[:, :3])

    voxel_input = tf.placeholder(tf.float32, [None, param.VOXEL_SHAPE[0], param.VOXEL_SHAPE[1], param.VOXEL_SHAPE[2], 1])

    with tf.Session() as sess:

        cnn_3d = Models.cnn_3d(voxel_input)

        saver = tf.train.Saver()
        saver.restore(sess, param.MODEL_DIR)

        objectness_predict = sess.run(cnn_3d.objectness_predict, feed_dict={voxel_input:voxel})
        print("objectness_pred : {}".format(objectness_predict))

    obj_array = MarkerArray()
    voxel_array = MarkerArray()
    obj_array = Utils.get_marker_array(objectness_predict, obj_array)
    voxel_array = Utils.get_marker_array(voxel, voxel_array, type="voxel")

    frame_id = 0
    while not rospy.is_shutdown():
        pc_pub.publish(points)
        obj_array_pub.publish(obj_array)
        voxel_array_pub.publish(voxel_array)
        print("Frame : %d" % frame_id)
        frame_id += 1
        sleep_rate.sleep()


def Visualize(filename):

    data_path = param.DATA_DIR + filename + ".pcd"
    calib_path = param.CALIB_DIR + filename + ".txt"
    label_path = param.LABEL_DIR + filename + ".txt"

    print "data path  : " + data_path
    print "calib path : " + calib_path
    print "label path : " + label_path

    pc_pub = rospy.Publisher("cnn_3d_points_raw", PointCloud2, queue_size=100000)
    rospy.init_node("cnn_3d")
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "cnn_3d"

    marker_array_pub = rospy.Publisher("cnn_3d_anchor_obj", MarkerArray, queue_size=1000)
    sleep_rate = rospy.Rate(1)

    point_cloud, gt_objectness = Loader.get_visualize_input(data_path, calib_path, label_path)
    points = pc2.create_cloud_xyz32(header, point_cloud[:, :3])

    marker_array = Utils.get_marker_array(gt_objectness, type="anchor")

    while not rospy.is_shutdown():
        pc_pub.publish(points)
        marker_array_pub.publish(marker_array)
        sleep_rate.sleep()


if __name__ == '__main__':

    Train()

    # Test("/media/vincent/DATA/Ubuntu/Project/Dataset/KITTI/VelodynePCD/debug/000729.pcd")

    # Visualize("000860")










