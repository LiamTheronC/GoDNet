import os
import numpy as np
from math import gcd
from numbers import Number
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import torch.optim as optim
import random
from metrics.metrics_waymo import MotionMetrics, output_metrics, _default_metrics_config, print_metrics
from utils import collate_fn, pre_gather
import time
import logging
from waymo_open_dataset.metrics.python import config_util_py as config_util

from model.laneGCN import GreatNet
from losses.lanegcn import Loss


class W_Dataset(Dataset):
    def __init__(self,path) -> None:

        self.path = path
        self.files = os.listdir(path)
    
    def __getitem__(self, index) -> dict:

        data_path = os.path.join(self.path,self.files[index])
        data = torch.load(data_path)

        return data
    
    def __len__(self) -> int:

        return len(self.files)


def val1(net, val_loader, loss_f, epoch, num_epochs, post):
    net.eval()
    metrics = dict()
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):

            outputs = net(data)
            loss_out = loss_f(outputs,data)
            post.append(metrics,loss_out,outputs,data)
    
    dt = time.time() - start_time
    _, Tfde = post.display(metrics, dt, epoch, num_epochs, "Validation")


def main():

    # seed = 33

    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    #/home/avt/prediction/Waymo/data_processed/vp/val_5f/2_282.pt

    config = dict()
    config['n_actornet'] = 128
    config['num_epochs'] = 50
    config['lr'] = 1e-3
    config["num_scales"] = 6
    config["n_map"] = 128
    config["n_actor"] = 128
    config['n_mark'] = 128
    config["actor2map_dist"] = 7.0
    config["map2actor_dist"] = 6.0
    config["actor2actor_dist"] = 50.0
    config["num_mods"] = 6
    config["pred_size"] = 80
    config["pred_step"] = 1
    config["num_preds"] = config["pred_size"] // config["pred_step"]
    config["cls_th"] = 2.0
    config["cls_th_2"] = 0.6
    config["cls_ignore"] = 0.2
    config["mgn"] = 0.2
    config["cls_coef"] = 1.0
    config["reg_coef"] = 1.0
    config["metrics_preds"] = 80
    config['acrs'] = [20,40,60]
    config['cut'] = range(10,50)
    config["dim_feats"] = {'xyvp':[6,2], 'xyz':[4,3], 'xy':[3,2], 'xyp':[4,2], 'vp':[4,2], 'vpt':[5,2]}
    config['type_feats'] = 'vp'
    config['f'] = '100f'
    config['name'] = 'laneGCN'
    
    config['val_final'] = '/home/avt/prediction/Waymo/data_processed/vp/val_final' 
    config['model_weights'] = 'weights/'+ config['name'] + '_' + config['type_feats'] + '_' + config['f'] + '0721.pth'

    net = GreatNet(config)
    net.load_state_dict(torch.load(config['model_weights']))
    net.cuda()

    loss_f = Loss(config)
    loss_f.cuda()
    
    batch_size = 4
    
    dataset_val = W_Dataset(config['val_final'])
    val_loader = DataLoader(dataset_val, 
                           batch_size = batch_size ,
                           collate_fn = collate_fn, 
                           shuffle = True, 
                           drop_last=True)

    num_epochs = config['num_epochs']

    log_file = 'validation.log'
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=log_file, level=logging.INFO, format=log_format)

    metrics_config = _default_metrics_config()
    motion_metrics = MotionMetrics(metrics_config)
    metric_names = config_util.get_breakdown_names_from_motion_config(metrics_config)

    net.eval()
    with torch.no_grad():
        for epoch in range(num_epochs):
            for batch_idx, data in enumerate(val_loader):
                out = net(data)

                pred_trajectory, pred_score, gt_trajectory, gt_is_valid, pred_gt_indices, \
                    pred_gt_indices_mask, object_type = output_metrics(out, data)
                
                motion_metrics.update_state(pred_trajectory, pred_score, gt_trajectory,
                              gt_is_valid, pred_gt_indices,
                              pred_gt_indices_mask, object_type)
                
            train_metric_values = motion_metrics.result()
            print_metrics(train_metric_values, metric_names)
            break

if __name__ == "__main__":
    main()
    

