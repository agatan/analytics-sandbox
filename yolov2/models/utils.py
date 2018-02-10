"""utils module provides utility layers for constructing networks.
"""

import tensorflow as tf
from typing import Union, Tuple

K = tf.keras
M = K.models
L = K.layers


def conv2d_bn_leaky(filters: int, kernel_size: Union[int, Tuple[int, int]]):
    """conv2d_bn_leaky is a composed layer that consists of a convolution layer,
    batch normalization, and leaky ReLu activation.

    Arguments
        filters: number of filters of convolution.
        kernel_size: kernel size of convolution.
    """
    def f(inputs):
        x = L.Conv2D(filters, kernel_size, padding='same',
                     use_bias=False)(inputs)
        x = L.BatchNormalization()(x)
        return L.LeakyReLU(alpha=0.1)(x)
    return f
