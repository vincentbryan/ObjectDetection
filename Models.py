import tensorflow as tf
import Parameter as param


class cnn_3d(object):
    def __init__(self, voxel):
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.object_detection = None
        self.objectness_predict = None
        self.boundary_box = None
        self.build_model(voxel)
        pass

    def conv_3d_layer(self, input_layer, input_dim, output_dim,
                      shape, stride, name="", activation=tf.nn.relu,
                      is_training=True, padding="SAME"):

        shape.append(input_dim)
        shape.append(output_dim)

        with tf.variable_scope("conv_3d"):
            kernel = tf.get_variable("weights",
                                     shape=shape,
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.001))
            bias = tf.get_variable("bias", shape=output_dim, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            output_layer = tf.nn.conv3d(input_layer, kernel, stride, padding=padding)
            output_layer = tf.nn.bias_add(output_layer, bias)

            output_layer = activation(output_layer, name=name)

            # TODO batch normalization!!!

        return output_layer

    def pool_3d_layer(self, input_layer, shape, name="", padding="VALID"):
        with tf.variable_scope("pool"):
            stride = [1, shape[0], shape[1], shape[2], 1]
            shape_ = [1, shape[0], shape[1], shape[2], 1]
            output_layer = tf.nn.max_pool3d(input_layer, shape_, stride, name=name, padding=padding)
            return output_layer

    def full_3d_layer(self, input_layer, input_dim, output_dim, name=""):
        with tf.variable_scope("full_3d"):
            kernel = tf.get_variable("weight",
                                     shape=[input_dim, output_dim],
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.01))
            output_layer = tf.matmul(tf.reshape(input_layer, [-1, input_dim]), kernel, name=name)
            return tf.reshape(output_layer, [-1, param.ANCHOR_SHAPE[0], param.ANCHOR_SHAPE[1], param.ANCHOR_SHAPE[2], output_dim])

    def output(self, input_layer, input_dim, output_dim, shape, stride, padding="SAME", name=""):

        shape.append(input_dim)
        shape.append(output_dim)

        with tf.variable_scope("output"):
            kernel = tf.get_variable("weights",
                                     shape=shape,
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.01))
            output_layer = tf.nn.conv3d(input_layer, kernel, stride, padding=padding, name=name)

        return output_layer

    def build_model(self, voxel):

        self.layer1 = self.conv_3d_layer(voxel, 1, 48, [4, 4, 4], [1, 4, 4, 4, 1], name="layer1")
        # self.layer2 = self.conv_3d_layer(self.layer1, 16, 32, [4, 4, 4], stride, name="layer2")
        self.layer2 = self.pool_3d_layer(self.layer1, [4, 4, 4], name="layer2")
        # self.layer3 = self.conv_3d_layer(self.layer2, 16, 64, [4, 4, 4], stride, name="layer3")
        self.layer3 = self.full_3d_layer(self.layer2, 48, 2, name="layer3")
        # self.layer4 = self.conv_3d_layer(self.layer3, 64, 64, [4, 4, 4], stride, name="layer4")

        # self.object_detection = self.output(self.layer4, 64, 2, [3, 3, 3], stride, name="object_detection")
        self.objectness_predict = tf.nn.softmax(self.layer3, axis=-1, name="obj_pred")



