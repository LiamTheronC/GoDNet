
import torch
import csv
import argparse
import torch.functional as F
from torch.utils.data import DataLoader
import time

import sys
from shapely.geometry import LineString
from shapely.affinity import affine_transform, rotate
from tqdm import tqdm

import tarfile

import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.models import model as model_utils
from mtr.utils import common_utils, motion_utils
from mtr.datasets import build_dataloader

from easydict import EasyDict

from waymo_open_dataset.protos import motion_submission_pb2

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--output_dir', type=str, default=None, help='output directory for submission')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def save_to_file(scnerio_predictions, args):
    print('saving....')
    submission = motion_submission_pb2.MotionChallengeSubmission(
        account_name='hzyqiang@gmail.com', unique_method_name='GTR-ens',
        authors=['Haochen Liu','Xiaoyu Mo','Zhiyu Huang','Chen Lv'], 
        affiliation='Nanyang Technological University', 
        submission_type=1, scenario_predictions=scnerio_predictions
        )

    save_path = args.output_dir + f"test_submission4.proto"
    tar_path = args.output_dir + f"test_submission4.tar.gz"
    f = open(save_path, "wb")
    f.write(submission.SerializeToString())
    f.close()

    print('Testing_saved,zipping...')

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(save_path)
        tar.close()

    print('Finished!')

def generate_prediction_dicts(batch_dict, output_path=None):
        """

        Args:
            batch_dict:
                pred_scores: (num_center_objects, num_modes)
                pred_trajs: (num_center_objects, num_modes, num_timestamps, 7)

              input_dict:
                center_objects_world: (num_center_objects, 10)
                center_objects_type: (num_center_objects)
                center_objects_id: (num_center_objects)
                center_gt_trajs_src: (num_center_objects, num_timestamps, 10)
        """
        input_dict = batch_dict['input_dict']

        pred_scores = batch_dict['pred_scores']
        pred_trajs = batch_dict['pred_trajs']
        center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)

        num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
        assert num_feat == 7

        pred_trajs_world = common_utils.rotate_points_along_z(
            points=pred_trajs.view(num_center_objects, num_modes * num_timestamps, num_feat),
            angle=center_objects_world[:, 6].view(num_center_objects)
        ).view(num_center_objects, num_modes, num_timestamps, num_feat)
        pred_trajs_world[:, :, :, 0:2] += center_objects_world[:, None, None, 0:2]

        pred_dict_list = []
        for obj_idx in range(num_center_objects):
            single_pred_dict = {
                'scenario_id': input_dict['scenario_id'][obj_idx],
                'pred_trajs': pred_trajs_world[obj_idx, :, :, 0:2].cpu().numpy(),
                'pred_scores': pred_scores[obj_idx, :].cpu().numpy(),
                'object_id': input_dict['center_objects_id'][obj_idx],
                'object_type': input_dict['center_objects_type'][obj_idx],
                'gt_trajs': input_dict['center_gt_trajs_src'][obj_idx].cpu().numpy(),
                'track_index_to_predict': input_dict['track_index_to_predict'][obj_idx].cpu().numpy()
            }
            pred_dict_list.append(single_pred_dict)

        return pred_dict_list

def traj_serialize(trajectories, scores, object_ids):
    scored_obj_trajs = []
    for i in range(trajectories.shape[0]):
        center_x, center_y = trajectories[i, 4::5, 0], trajectories[i, 4::5, 1]
        traj = motion_submission_pb2.Trajectory(center_x=center_x, center_y=center_y)
        object_traj = motion_submission_pb2.ScoredTrajectory(confidence=scores[i], trajectory=traj)
        scored_obj_trajs.append(object_traj)
    return motion_submission_pb2.SingleObjectPrediction(trajectories=scored_obj_trajs, object_id=object_ids)

def serialize_single_scenario(scenario_list):
    single_prediction_list = []
    scenario_id = scenario_list[0]['scenario_id']
    for single_dict in scenario_list:
        sc_id = single_dict['scenario_id']
        assert sc_id == scenario_id
        single_prediction = traj_serialize(single_dict['pred_trajs'],single_dict['pred_scores'], single_dict['object_id'])
        single_prediction_list.append(single_prediction)
    prediction_set = motion_submission_pb2.PredictionSet(predictions=single_prediction_list)
    return motion_submission_pb2.ChallengeScenarioPredictions(scenario_id=scenario_id, single_predictions=prediction_set)

def serialize_single_batch(final_pred_dicts, batch_sample_count, scenario_predictions):
    i = 0
    for pred_objects_per_sc in batch_sample_count:
        scenario_list = []
        for _ in range(pred_objects_per_sc):
            scenario_list.append(final_pred_dicts[i])
            i += 1
        scenario_preds = serialize_single_scenario(scenario_list)
        scenario_predictions.append(scenario_preds)


def test(model, test_data, args):
    scenario_predictions = []
    size = len(test_data)*args.batch_size
    with torch.no_grad():
        for i, batch_dict in enumerate(test_data):

            batch_pred_dicts = model(batch_dict)
            final_pred_dicts = generate_prediction_dicts(batch_pred_dicts)
            serialize_single_batch(final_pred_dicts, batch_dict['batch_sample_count'], scenario_predictions)
            
            sys.stdout.write(f'\rProcessed:{i*args.batch_size}-{size}')
            sys.stdout.flush()
        
        save_to_file(scenario_predictions, args)

def main():
    args, cfg = parse_config()

    eval_output_dir = '/home/users/ntu/haochen0/scratch/waymo23/sub_res/'
    log_file = eval_output_dir + f"log_test3.txt"#_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        dist=False, workers=args.workers, logger=logger, training=False,
        submission=True,testing=True
    )

    cfg_from_yaml_file(cfg, CFG)
    model = model_utils.MotionTransformer(config=CFG.MODEL)
    it, epoch = model.load_params_from_file(filename=kpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    test(model, test_loader, args)


if __name__ == '__main__':
    main()
