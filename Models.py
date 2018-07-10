import tensorflow as tf
import Parameter as param


def cnn_3d(sess, voxel, is_training=True):

    phase = tf.placeholder(tf.bool, name='phase') if is_training else None

    with tf.variable_scope("cnn_3d") as scope:
        model = Layers()
        model.build_model(voxel)

    # if is_training:
    #     initialized_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="cnn_3d")
    #     sess.run(tf.variables_initializer(initialized_var))

    return model


class Layers(object):
    def __init__(self):
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.object_detection = None
        self.objectness_predict = None
        self.boundary_box = None
        pass

    def conv_3d_layer(self, input_layer, input_dim, output_dim,
                      shape, stride, name="", activation=tf.nn.relu,
                      is_training=True, padding="SAME"):

        shape.append(input_dim)
        shape.append(output_dim)

        with tf.variable_scope("conv_3d/" + name):
            kernel = tf.get_variable("weights",
                                     shape=shape,
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.001))
            bias = tf.get_variable("bias", shape=output_dim, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            output_layer = tf.nn.conv3d(input_layer, kernel, stride, padding=padding)
            output_layer = tf.nn.bias_add(output_layer, bias)

            output_layer = activation(output_layer, name="activation")

            # TODO batch normalization!!!
            # output_layer = activation(output_layer, name="activation") if activation else batch_norm(output_layer, is_training)

        return output_layer

    def output(self, input_layer, input_dim, output_dim, shape, stride, padding="SAME", name=""):

        shape.append(input_dim)
        shape.append(output_dim)

        with tf.variable_scope("conv_3d/" + name):
            kernel = tf.get_variable("weights",
                                     shape=shape,
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.01))
            output_layer = tf.nn.conv3d(input_layer, kernel, stride, padding=padding)
        return output_layer

    def build_model(self, voxel):

        stride_step = param.STRIDE_STEP
        stride = [1, stride_step, stride_step, stride_step, 1]

        self.layer1 = self.conv_3d_layer(voxel, 1, 16, [5, 5, 5], stride, name="layer1")
        self.layer2 = self.conv_3d_layer(self.layer1, 16, 32, [5, 5, 5], stride, name="layer2")
        self.layer3 = self.conv_3d_layer(self.layer2, 32, 64, [3, 3, 3], stride, name="layer3")
        self.layer4 = self.conv_3d_layer(self.layer3, 64, 64, [3, 3, 3], stride, name="layer4")

        stride = [1, 1, 1, 1, 1]
        self.object_detection = self.output(self.layer4, 64, 2, [3, 3, 3], stride, name="object_detection")
        self.objectness_predict = tf.nn.softmax(self.object_detection, axis=-1, name="obj_pred")



