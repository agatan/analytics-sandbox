"""darknet module provides darknet19 network construction functions.
"""

from typing import Tuple, Union

import tensorflow as tf

from .utils import conv2d_bn_leaky


def darknet19(inputs: tf.Tensor) -> tf.Tensor:
    """Construct darknet19 network.

    Arguments
        inputs: input tensor with shape (batch_size, width, height, channel)
    """
    x = conv2d_bn_leaky(inputs, 32, 3)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = conv2d_bn_leaky(x, 64, 3)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = conv2d_bn_leaky(x, 128, 3)
    x = conv2d_bn_leaky(x, 64, 1)
    x = conv2d_bn_leaky(x, 128, 3)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = conv2d_bn_leaky(x, 256, 3)
    x = conv2d_bn_leaky(x, 128, 1)
    x = conv2d_bn_leaky(x, 256, 3)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = conv2d_bn_leaky(x, 512, 3)
    x = conv2d_bn_leaky(x, 256, 1)
    x = conv2d_bn_leaky(x, 512, 3)
    x = conv2d_bn_leaky(x, 256, 1)
    conv13 = x = conv2d_bn_leaky(x, 512, 3)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = conv2d_bn_leaky(x, 1024, 3)
    x = conv2d_bn_leaky(x, 512, 1)
    x = conv2d_bn_leaky(x, 1024, 3)
    x = conv2d_bn_leaky(x, 512, 1)
    x = conv2d_bn_leaky(x, 1024, 3)
    x = tf.layers.conv2d(x, 1000, kernel_size=1,
                              padding='same', activation=tf.nn.softmax)
    return conv13, x


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, shape=(None, 416, 416, 3))
    conv13, outputs = darknet19(inputs)
    print(outputs)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('/tmp/log', sess.graph)
        import numpy as np
        sess.run([outputs, conv13], feed_dict={inputs: np.zeros((1, 416,416,3),dtype=np.float32)})
        writer.close()
