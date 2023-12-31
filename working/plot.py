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
from memory_profiler import profile
from metrics.metrics import Postprocess
from utils import collate_fn, pre_gather
import matplotlib.pyplot as plt
import time

from model.GANet import GreatNet
from losses.ganet import Loss


class W_Dataset(Dataset):
    def __init__(self,path) -> None:

        self.path = path
        self.files = os.listdir(path)
    
    def __getitem__(self, index) -> dict:

        data_path = os.path.join(self.path,self.files[index])
        data = torch.load(data_path)
        print(data_path)

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
    config['name'] = 'GoDNet'
    config['train_split'] = '/home/avt/prediction/Waymo/data_processed/' + config['type_feats'] + '/train_' + config['f'] 
    config['val_split'] = '/home/avt/prediction/Waymo/data_processed/' + config['type_feats'] + '/val_' + '5f'
    config['plot'] = '/home/avt/prediction/Waymo/data_processed/plot'
    config['model_weights'] = 'weights/'+ config['name'] + '_' + config['type_feats'] + '_' + config['f'] + '0719.pth'

    net = GreatNet(config)
    net.load_state_dict(torch.load(config['model_weights']))
    net.cuda()

    loss_f = Loss(config)
    loss_f.cuda()

    post = Postprocess(config)
    
    
    batch_size = 1
    
    dataset_val = W_Dataset(config['plot'])
    val_loader = DataLoader(dataset_val, 
                           batch_size = batch_size ,
                           collate_fn = collate_fn, 
                           shuffle = True, 
                           drop_last=True)

    num_epochs = config['num_epochs']

    net.eval()
    metrics = dict()
    start_time = time.time()
    with torch.no_grad():
        for epoch in range(num_epochs):
            
            # val1(net,val_loader,loss_f,epoch,num_epochs, post)
            # break

            for batch_idx, data in enumerate(val_loader):
                outputs = net(data)
                loss_out = loss_f(outputs,data)
                post.append(metrics,loss_out,outputs,data)
                msg,_ = post.display(metrics, 0, epoch, num_epochs, "Validation")
                post.plot(metrics, data, outputs, msg, 6, False, 1)
                break
            break

if __name__ == "__main__":
    main()
    

