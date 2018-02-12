"""
Simple example to clearify how to preprocess dataset, train a YOLO v2 model and do some predictions.

This program is based on https://github.com/allanzelener/YAD2K.
"""

from models.yolo import YOLO

import tensorflow as tf
import numpy as np
from PIL import Image


def main():
    # Instantiate YOLO v2 model.
    model = YOLO(image_shape=(416, 416, 3), classes=['octcat'])

    # Preprocess the target image.
    orig_image = Image.open('images/sample.png')
    orig_size = np.array([orig_image.width, orig_image.height])

    image = orig_image.resize((416, 416), Image.BICUBIC)
    image_data = np.array(image, dtype='float32') / 255.

    # Target boxes.
    # Each base box is in form of [x_min, y_min, x_max, y_max, class label].
    base_boxes = np.array([[120, 15, 280, 230, 0]])
    boxes_xy = 0.5 * (base_boxes[:, 0:2] + base_boxes[:, 2:4]) / orig_size
    boxes_wh = (base_boxes[:, 2:4] - base_boxes[:, 0:2]) / orig_size
    # Each box is in form of [y_center, x_center, height, width, class label].
    boxes = np.concatenate((boxes_xy, boxes_wh, base_boxes[:, -1:]), axis=1)
    detectors_mask, matching_true_boxes = model.preprocess_true_boxes(boxes)

    # Expand arrays to construct min-batch (batch size is 1 here).
    image_data_expanded = np.expand_dims(image_data, axis=0)
    boxes_expanded = np.expand_dims(boxes, axis=0)
    detectors_mask_expanded = np.expand_dims(detectors_mask, axis=0)
    matching_true_boxes_expanded = np.expand_dims(matching_true_boxes, axis=0)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        model.train(sess, image_data_expanded, boxes_expanded,
                    detectors_mask_expanded, matching_true_boxes_expanded)
        model.save(sess, '/tmp/yolo', 'yolo')


if __name__ == '__main__':
    main()
