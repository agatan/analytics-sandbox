"""darknet module provides the darknet19 network construction function.
"""

import tensorflow as tf
from .utils import conv2d_bn_leaky

K = tf.keras
M = K.models
L = K.layers


def darknet19(inputs):
    """Construct darknet19 network.

    Arguments
        inputs: input tensor with shape (batch_size, width, height, channel)
    """
    x = conv2d_bn_leaky(32, 3)(inputs)
    x = L.MaxPooling2D()(x)
    x = conv2d_bn_leaky(64, 3)(x)
    x = L.MaxPooling2D()(x)
    x = conv2d_bn_leaky(128, 3)(x)
    x = conv2d_bn_leaky(64, 1)(x)
    x = conv2d_bn_leaky(128, 3)(x)
    x = L.MaxPooling2D()(x)
    x = conv2d_bn_leaky(256, 3)(x)
    x = conv2d_bn_leaky(128, 1)(x)
    x = conv2d_bn_leaky(256, 3)(x)
    x = L.MaxPooling2D()(x)
    x = conv2d_bn_leaky(512, 3)(x)
    x = conv2d_bn_leaky(256, 1)(x)
    x = conv2d_bn_leaky(512, 3)(x)
    x = conv2d_bn_leaky(256, 1)(x)
    x = conv2d_bn_leaky(512, 3)(x)
    x = L.MaxPooling2D()(x)
    x = conv2d_bn_leaky(1024, 3)(x)
    x = conv2d_bn_leaky(512, 1)(x)
    x = conv2d_bn_leaky(1024, 3)(x)
    x = conv2d_bn_leaky(512, 1)(x)
    x = conv2d_bn_leaky(1024, 3)(x)
    logits = L.Conv2D(1000, 1, padding='same', activation='softmax')(x)
    return logits


def print_summary():
    """Print network summary of darknet19
    """
    inputs = L.Input(shape=(224, 224, 3))
    outputs = darknet19(inputs)
    model = M.Model(inputs, outputs)
    model.summary()


if __name__ == '__main__':
    print_summary()
