from .darknet import darknet19
from .utils import conv2d_bn_leaky

import tensorflow as tf
import numpy as np

from typing import Tuple, List

K = tf.keras
L = K.layers
M = K.models


YOLO_ANCHORS = np.array([
    [1.08, 1.19],
    [3.42, 4.41],
    [6.63, 11.38],
    [9.42, 5.11],
    [16.62, 10.52]], dtype='float32')


class YOLO():
    def __init__(self, image_shape: Tuple[int, int, int],
                 classes: List[str],
                 anchors: np.ndarray = YOLO_ANCHORS):
        self.image_shape = image_shape
        self.anchors = anchors
        self.classes = classes

        self._build_model()
        self._build_loss_model()

    @property
    def n_anchors(self):
        return len(self.anchors)

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def detectors_mask_shape(self):
        w, h, _ = self.image_shape
        return (w / 32, h / 32, self.n_anchors, 1)

    @property
    def matching_boxes_shape(self):
        w, h, _ = self.image_shape
        return (w / 32, h / 32, self.n_anchors, self.n_anchors)

    def _build_model(self):
        """Create YOLO v2 model CNN body.

        Argument
            inputs: input tensor
            n_anchors: number of bouding box anchors
            n_classes: number of classes
        """
        inputs = L.Input(shape=self.image_shape)
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
        # Tha last FCN layer (= extracted image features for 1/32 scale)
        image_features = L.Conv2D(self.n_anchors * (self.n_classes + 5),
                                  1, padding='same')(x)

        outputs = self._extract_bounding_boxes_layer(image_features)
        self.model = M.Model(inputs, outputs)
        return self.model

    def _build_loss_model(self):
        boxes_input = L.Input(shape=(None, 5))
        detectors_mask_input = L.Input(shape=self.detectors_mask_shape)
        matching_boxes_input = L.Input(shape=self.matching_boxes_shape)
        loss_layer = L.Lambda(
            _yolo_loss_function,
            output_shape=(1,),
            name='yolo_loss',
            arguments={
                'anchors': self.anchors,
                'n_classes': self.n_classes,
            })
        loss_output = loss_layer([
            *self.model.output,
            boxes_input, detectors_mask_input, matching_boxes_input,
        ])
        self.model_loss = M.Model(
            [self.model.input, boxes_input, detectors_mask_input,
             matching_boxes_input], loss_output)
        self.model_loss.compile(optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })

    def _extract_bounding_boxes_layer(self, image_features):
        """Convert final layer image features to bounding box parameters.

        Args:
            feats: Final convolutional layer features.
            anchors: Anchor box widths and heights.
            n_classes: Number of target classes.

        Returns:
            image_features (tf.Tensor): Raw image features
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

        conv_dims = image_features.shape[1:3]
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
        conv_index = tf.cast(conv_index, image_features.dtype)

        def reshape_lambda(x):
            return tf.reshape(
                x, shape=[-1, conv_dims[0], conv_dims[1],
                          self.n_anchors, self.n_classes + 5])

        image_features_reshaped = L.Lambda(
            reshape_lambda, name='image_features')(image_features)
        conv_dims_reshaped = tf.cast(tf.reshape(
            conv_dims, [1, 1, 1, 1, 2]), image_features_reshaped.dtype)

        box_confidence = L.Lambda(lambda x: tf.sigmoid(
            x[..., 4:5]), name='box_confidence')(image_features_reshaped)
        box_class_probs = L.Lambda(lambda x: tf.nn.softmax(
            x[..., 5:]), name='box_class_probs')(image_features_reshaped)
        box_xy = L.Lambda(lambda x:
                          (tf.sigmoid(x[..., :2]) +
                           conv_index) / conv_dims_reshaped,
                          name='box_xy',
                          )(image_features_reshaped)
        box_wh = L.Lambda(lambda x:
                          tf.exp(x[..., 2:4]) * anchors_tensor /
                          conv_dims_reshaped,
                          name='box_wh',
                          )(image_features_reshaped)

        return image_features_reshaped, box_xy, box_wh, box_confidence, box_class_probs

    def summary(self):
        print("=====> Prediction Model")
        self.model.summary()

        print()
        print("=====> Loss Model")
        self.model_loss.summary()


def _yolo_loss_function(args, anchors, n_classes):
    image_features, pred_xy, pred_wh, pred_confidence, pred_class_prob, \
        true_boxes, detectors_mask, matching_true_boxes = args

    pred_boxes = tf.concat(
        (tf.sigmoid(image_features[..., 0:2]), image_features[..., 2:4]),
        axis=-1)

    pred_xy = tf.expand_dims(pred_xy, 4)
    pred_wh = tf.expand_dims(pred_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    true_boxes_shape = tf.shape(true_boxes)

    true_boxes = tf.reshape(true_boxes, [
        true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
    ])
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    best_ious = K.backend.max(iou_scores, axis=4)
    best_ious = K.backend.expand_dims(best_ious)

    object_detections = tf.cast(best_ious > 0.6, best_ious.dtype)

    no_object_scale = 1
    no_object_weights = no_object_scale * \
        (1 - object_detections) * (1 - detectors_mask)
    no_objects_loss = no_object_weights * tf.square(pred_confidence)
    objects_loss = 5 * detectors_mask * tf.square(1 - pred_confidence)
    confidence_loss = objects_loss + no_objects_loss

    matching_classes = tf.one_hot(
        tf.cast(matching_true_boxes[..., 4], tf.int32), n_classes)
    classification_loss = 1 * detectors_mask * \
        tf.square(matching_classes - pred_class_prob)
    matching_boxes = matching_true_boxes[..., 0:4]
    coordinates_loss = 1 * detectors_mask * \
        tf.square(matching_boxes - pred_boxes)

    confidence_loss_sum = K.backend.sum(confidence_loss)
    classification_loss_sum = K.backend.sum(classification_loss)
    coordinates_loss_sum = K.backend.sum(coordinates_loss)
    total_loss = 0.5 * (confidence_loss_sum +
                        classification_loss_sum + coordinates_loss_sum)

    tf.Print(total_loss,
             [total_loss, confidence_loss_sum, classification_loss_sum,
              coordinates_loss_sum],
             message='yolo_loss, confidence_loss, class_loss, box_coord_loss:')

    return total_loss


def print_summary():
    """Print network summary of YOLO v2
    """
    model = YOLO(
        image_shape=(416, 416, 3),
        classes=[str(i) for i in range(20)],
    )
    model.summary()


if __name__ == '__main__':
    print_summary()
