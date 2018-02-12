"""darknet module provides darknet19 network construction functions.
"""

from typing import Tuple, Union

import tensorflow as tf


def conv2d_bn_leaky(x: tf.Tensor, filters: int, kernel_size: Union[int, Tuple[int, int]]):
    """conv2d_bn_leaky is a composed layer that consists of a convolution layer,
    batch normalization, and leaky ReLu activation.

    Arguments
        x: input tensor
        filters: number of filters of convolution.
        kernel_size: kernel size of convolution.
    """
    x = tf.layers.conv2d(x, filters, kernel_size=kernel_size, padding='same',
                         use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(5e-4))
    x = tf.layers.batch_normalization(x)
    return tf.keras.layers.LeakyReLU(alpha=0.1)(x)


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
    x = conv2d_bn_leaky(x, 512, 3)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = conv2d_bn_leaky(x, 1024, 3)
    x = conv2d_bn_leaky(x, 512, 1)
    x = conv2d_bn_leaky(x, 1024, 3)
    x = conv2d_bn_leaky(x, 512, 1)
    x = conv2d_bn_leaky(x, 1024, 3)
    logits = tf.layers.conv2d(x, 1000, kernel_size=1,
                              padding='same', activation=tf.nn.softmax)
    return logits


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, shape=(None, 416, 416, 3))
    outputs = darknet19(inputs)
    print(outputs)
