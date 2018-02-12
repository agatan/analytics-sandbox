import tensorflow as tf
import numpy as np

from models.yolo import YOLO

class YOLOTrainer:
    """YOLOTrainer is responsible to run YOLO model training, save parameters,
    and summarize training process for tensorboard.
    """

    def __init__(self, sess: tf.Session, model: YOLO, summary_dir: str):
        self.sess = sess
        self.model = model

        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(self.init_op)

        tf.summary.scalar('loss', self.model.loss)
        self.merged_summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(summary_dir)
        self.summary_writer.add_graph(sess.graph)

    def train_step(self, image_data: np.ndarray, gt_boxes: np.ndarray, gt_detectors_mask: np.ndarray, gt_matching_boxes: np.ndarray):
        loss, _, summary = self.sess.run([self.model.loss, self.model.train_op, self.merged_summary], feed_dict={
            self.model.inputs: image_data,
            self.model.gt_boxes: gt_boxes,
            self.model.gt_detectors_mask: gt_detectors_mask,
            self.model.gt_matching_boxes: gt_matching_boxes,
        })
        cur_step = self.model.global_step_tensor.eval(self.sess)
        self.summary_writer.add_summary(summary, cur_step)
