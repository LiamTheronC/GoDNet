#not yet working

import sys
sys.path.append('/home/avt/prediction/Waymo/working/')

import os
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.protos import motion_metrics_pb2
from waymo_open_dataset.metrics.python import config_util_py as config_util


def _default_metrics_config():
  config = motion_metrics_pb2.MotionMetricsConfig()
  config_text = """
  track_steps_per_second: 10
  prediction_steps_per_second: 2
  track_history_samples: 10
  track_future_samples: 80
  speed_lower_bound: 1.4
  speed_upper_bound: 11.0
  speed_scale_lower: 0.5
  speed_scale_upper: 1.0
  step_configurations {
    measurement_step: 5
    lateral_miss_threshold: 1.0
    longitudinal_miss_threshold: 2.0
  }
  step_configurations {
    measurement_step: 9
    lateral_miss_threshold: 1.8
    longitudinal_miss_threshold: 3.6
  }
  step_configurations {
    measurement_step: 15
    lateral_miss_threshold: 3.0
    longitudinal_miss_threshold: 6.0
  }
  max_predictions: 6
  """
  text_format.Parse(config_text, config)
  return config


class MotionMetrics(tf.keras.metrics.Metric):
  """Wrapper for motion metrics computation."""

  def __init__(self, config):
    super().__init__()
    self._prediction_trajectory = []
    self._prediction_score = []
    self._ground_truth_trajectory = []
    self._ground_truth_is_valid = []
    self._prediction_ground_truth_indices = []
    self._prediction_ground_truth_indices_mask = []
    self._object_type = []
    self._metrics_config = config

  def reset_state(self):
    self._prediction_trajectory = []
    self._prediction_score = []
    self._ground_truth_trajectory = []
    self._ground_truth_is_valid = []
    self._prediction_ground_truth_indices = []
    self._prediction_ground_truth_indices_mask = []
    self._object_type = []

  def update_state(self, prediction_trajectory, prediction_score,
                   ground_truth_trajectory, ground_truth_is_valid,
                   prediction_ground_truth_indices,
                   prediction_ground_truth_indices_mask, object_type):
    
    self._prediction_trajectory.append(prediction_trajectory)
    self._prediction_score.append(prediction_score)
    self._ground_truth_trajectory.append(ground_truth_trajectory)
    self._ground_truth_is_valid.append(ground_truth_is_valid)
    self._prediction_ground_truth_indices.append(
        prediction_ground_truth_indices)
    self._prediction_ground_truth_indices_mask.append(
        prediction_ground_truth_indices_mask)
    self._object_type.append(object_type)

  def result(self):
    # [batch_size, num_preds, 1, 1, steps, 2].
    # The ones indicate top_k = 1, num_agents_per_joint_prediction = 1.
    prediction_trajectory = tf.concat(self._prediction_trajectory, 0)
    # [batch_size, num_preds, 1].
    prediction_score = tf.concat(self._prediction_score, 0)
    # [batch_size, num_agents, gt_steps, 7].
    ground_truth_trajectory = tf.concat(self._ground_truth_trajectory, 0)
    # [batch_size, num_agents, gt_steps].
    ground_truth_is_valid = tf.concat(self._ground_truth_is_valid, 0)
    # [batch_size, num_preds, 1].
    prediction_ground_truth_indices = tf.concat(
        self._prediction_ground_truth_indices, 0)
    # [batch_size, num_preds, 1].
    prediction_ground_truth_indices_mask = tf.concat(
        self._prediction_ground_truth_indices_mask, 0)
    # [batch_size, num_agents].
    object_type = tf.cast(tf.concat(self._object_type, 0), tf.int64)

    # We are predicting more steps than needed by the eval code. Subsample.
    interval = (
        self._metrics_config.track_steps_per_second //
        self._metrics_config.prediction_steps_per_second)
    
    prediction_trajectory = prediction_trajectory[...,
                                                  (interval - 1)::interval, :]

    return py_metrics_ops.motion_metrics(
        config=self._metrics_config.SerializeToString(),
        prediction_trajectory=prediction_trajectory,
        prediction_score=prediction_score,
        ground_truth_trajectory=ground_truth_trajectory,
        ground_truth_is_valid=ground_truth_is_valid,
        prediction_ground_truth_indices=prediction_ground_truth_indices,
        prediction_ground_truth_indices_mask=prediction_ground_truth_indices_mask,
        object_type=object_type)
  

def output_metrics(out,data):

    # prediction, score
    reg = out['reg']
    cls = out['cls']
    indx = data['target_indx_e']

    batch_size = len(reg)

    preds = [reg[i][indx[i]] for i in range(batch_size)]
    scores = [cls[i][indx[i]] for i in range(batch_size)]

    preds = torch.concatenate(preds,0)
    scores = torch.concatenate(scores,0)

    #ground truth, has_gt
    trajs_xyz = data['trajs_xyz']
    v_h = data['velocity_xy_heading']
    shapes = data['shapes']
    indx_ = data['target_indx']
    valid = data['valid_masks']

    trajs_xy = [torch.stack(trajs_xyz[i],0)[indx_[i],:,:2] for i in range(batch_size)]
    l_w = [torch.stack(shapes[i],0)[indx_[i],:2] for i in range(batch_size)]
    y = [torch.stack(v_h[i],0)[indx_[i],:,2] for i in range(batch_size)]
    v = [torch.stack(v_h[i],0)[indx_[i],:,:2] for i in range(batch_size)]
    has = [torch.stack(valid [i],0)[indx_[i]] for i in range(batch_size)]
    
    trajs_xy = torch.concatenate(trajs_xy, 0)
    l_w = torch.concatenate(l_w, 0).unsqueeze(1).repeat(1,91,1)
    y = torch.concatenate(y,0).unsqueeze(-1)
    v = torch.concatenate(v,0)
    has = torch.concatenate(has,0)

    gt = torch.concatenate([trajs_xy, l_w , y, v], -1)

    # gt_indx, gt_mask

    gt_indx = torch.tensor(range(1)).repeat(gt.size(0))
    gt_mask = torch.tensor([True]).repeat(gt.size(0))

    # type

    types_dict = dict()
    types_dict['TYPE_VEHICLE'] = 1.0
    types_dict['TYPE_PEDESTRIAN'] = 2.0
    types_dict['TYPE_CYCLIST'] = 3.0

    target_types = data['target_type']
    types_ = torch.tensor([types_dict[e] for f in target_types for e in f ])

    # formation

    #[B,1,6,1,80,2]
    preds = preds.unsqueeze(2).unsqueeze(1).to(torch.float32)

    #[B,1,6]
    scores = scores.unsqueeze(1).to(torch.float32)

    #[B,1,91,7]
    gt = gt.unsqueeze(1).to(torch.float32)

    #[B,1,91]
    has = has.unsqueeze(1)

    #[B,1,1]
    gt_indx = gt_indx.unsqueeze(1).unsqueeze(1)

    #[B,1,1]
    gt_mask = gt_mask.unsqueeze(1).unsqueeze(1)

    #[B,1]
    types_ = types_.unsqueeze(1).to(torch.float32)

    output = [preds, scores, gt, has, gt_indx, gt_mask, types_ ]

    output_ = []
    for xx in output:
        xx = tf.convert_to_tensor(xx.cpu().detach().numpy())
        output_.append(xx)

    return output_


def print_metrics(train_metric_values, metric_names):
  final_msgs = []

  for i, m in enumerate(
    ['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):

    results = []

    for j, n in enumerate(metric_names):
        msg = '{}/{}: {}'.format(m, n, train_metric_values[i, j])
        results.append(train_metric_values[i, j])
        logging.info(msg)
        print(msg)
    
    ave = np.asarray(results).mean().astype(np.float32)
    msg_ = '{}:{}'.format(m,ave)
    logging.info(msg_)
    final_msgs.append(msg_)

  print(final_msgs)

