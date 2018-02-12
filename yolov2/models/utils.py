"""utils module provides utility layers for constructing networks.
"""

import tensorflow as tf
from typing import Union, Tuple


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
