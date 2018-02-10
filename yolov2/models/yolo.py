from .darknet import darknet19
from .utils import conv2d_bn_leaky

import tensorflow as tf
import numpy as np

from typing import Tuple, List

K = tf.keras
L = K.layers
M = K.models


class YOLO():
    def __init__(self, input_shape: Tuple[int, int],
                 anchors: np.ndarray, classes: List[str]):
        self.input_shape = input_shape
        self.anchors = anchors
        self.classes = classes

        self._build_model()

    @property
    def n_anchors(self):
        return len(self.anchors)

    @property
    def n_classes(self):
        return len(self.classes)

    def _build_model(self):
        """Create YOLO v2 model CNN body.

        Argument
            inputs: input tensor
            n_anchors: number of bouding box anchors
            n_classes: number of classes
        """
        inputs = L.Input(shape=self.input_shape)
        darknet_model = M.Model(inputs, darknet19(inputs))
        conv20 = conv2d_bn_leaky(1024, 3)(darknet_model.output)
        conv20 = conv2d_bn_leaky(1024, 3)(conv20)

        conv13 = darknet_model.layers[43].output
        conv21 = conv2d_bn_leaky(64, 1)(conv13)
        conv21_reshaped = L.Lambda(
            lambda x: tf.space_to_depth(x, block_size=2),
            name='space_to_depth',
        )(conv21)

        x = L.concatenate([conv21_reshaped, conv20])
        x = conv2d_bn_leaky(1024, 3)(x)
        x = L.Conv2D(self.n_anchors * (self.n_classes + 5),
                     1, padding='same')(x)

        self.model = M.Model(inputs, x)
        return self.model

    def summary(self):
        self.model.summary()

    def postprocessing(self, feats: tf.Tensor):
        """Convert final layer image features to bounding box parameters.

        Args:
            feats: Final convolutional layer features.
            anchors: Anchor box widths and heights.
            n_classes: Number of target classes.

        Returns:
            box_xy (tf.Tensor): x, y box predictions adjusted by spatial
                                location in conv layer.
            box_wh (tf.Tensor): w, h box predictions adjusted by anchors and
                                conv spatial resolution.
            box_confidence (tf.Tensor): Probability estimate for whether each
                                        box contains any object.
            box_class_probs (tf.Tensor): Probability estimate for each box over
                                         class labels.
        """
        anchors_tensor = tf.reshape(tf.Variable(self.anchors), [
                                    1, 1, 1, self.n_anchors, 2])

        conv_dims = tf.shape(feats)[1:3]
        conv_height_index = K.backend.arange(0, stop=conv_dims[0])
        conv_width_index = K.backend.arange(0, stop=conv_dims[1])
        conv_height_index = tf.tile(conv_height_index, [conv_dims[1]])

        conv_width_index = tf.tile(tf.expand_dims(
            conv_width_index, 0), [conv_dims[0], 1])
        conv_width_index = K.backend.flatten(tf.transpose(conv_width_index))
        conv_index = tf.transpose(
            tf.stack([conv_height_index, conv_width_index]))
        conv_index = tf.reshape(
            conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
        conv_index = tf.cast(conv_index, K.backend.dtype(feats))

        feats = tf.reshape(
            feats, [-1, conv_dims[0], conv_dims[1], self.n_anchors, self.n_classes + 5])
        conv_dims = tf.cast(tf.reshape(
            conv_dims, [1, 1, 1, 1, 2]), K.backend.dtype(feats))

        box_xy = tf.sigmoid(feats[..., :2])
        box_wh = tf.exp(feats[..., 2:4])
        box_confidence = tf.sigmoid(feats[..., 4:5])
        box_class_probs = tf.nn.softmax(feats[..., 5:])

        box_xy = (box_xy + conv_index) / conv_dims
        box_wh = box_wh * anchors_tensor / conv_dims

        return box_xy, box_wh, box_confidence, box_class_probs


def print_summary():
    """Print network summary of YOLO v2
    """
    sample_anchors = np.array([
        [1.08, 1.19],
        [3.42, 4.41],
        [6.63, 11.38],
        [9.42, 5.11],
        [16.62, 10.52]], dtype=np.float32)
    model = YOLO(input_shape=(224, 224, 3), anchors=sample_anchors,
                 classes=[str(i) for i in range(20)])
    model.summary()


if __name__ == '__main__':
    print_summary()
