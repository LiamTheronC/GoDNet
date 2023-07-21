# M6 training

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import torch.optim as optim
import random
from utils import collate_fn
import logging
from memory_profiler import profile
from metrics.metrics import Postprocess
import time
from datetime import date

from model.laneGCN import GreatNet # GANet, laneGCN
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


def train1(net,train_loader,loss_f,optimizer,epoch,num_epochs, post):
    # output of the net is directly trajectory
    net.train()
    metrics = dict()
    start_time = time.time()
    for batch_idx, data in enumerate(train_loader):

        outputs = net(data)
        loss_out = loss_f(outputs,data)
        

        # Backward and optimize
        optimizer.zero_grad()
        loss_out.backward()
        optimizer.step()

        post.append(metrics,loss_out.item(),outputs,data)

    dt = time.time() - start_time
    post.display(metrics, dt, epoch, num_epochs, "Train")


def val1(net, val_loader, loss_f, epoch, num_epochs, post, Tfde_a, config):
    net.eval()
    metrics = dict()
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):

            outputs = net(data)
            loss_out = loss_f(outputs,data)
            post.append(metrics,loss_out.item(),outputs,data)
    
    dt = time.time() - start_time
    _, Tfde = post.display(metrics, dt, epoch, num_epochs, "Validation")

    if Tfde < Tfde_a:
        print('update weights')
        torch.save(net.state_dict(), 'weights/'+ config['name'] +'_'+ config['type_feats'] + '_' + config['f'] + config['dd'] +'.pth')
        Tfde_a = Tfde

    return Tfde_a


@profile
def main():
    seed = 33

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = dict()
    config['n_actornet'] = 128
    config['num_epochs'] = 150
    config['lr'] = 1e-3
    config["num_scales"] = 6
    config["n_map"] = 128
    config["n_actor"] = 128
    config['n_mark'] = 128
    config["actor2map_dist"] = 7.0 # 7.0
    config["map2actor_dist"] = 6.0 # 6.0
    config["actor2actor_dist"] = 50.0
    config["num_mods"] = 6
    config["pred_size"] = 80
    config["pred_step"] = 1
    config["num_preds"] = config["pred_size"] // config["pred_step"]
    config["cls_th"] = 2.0 #5.0, 2.0
    config["cls_th_2"] = 0.6
    config["cls_ignore"] = 0.2
    config["mgn"] = 0.2
    config["cls_coef"] = 1.0
    config["reg_coef"] = 1.0
    config["metrics_preds"] = 80
    config['acrs'] = [20,40,60] # [40,80]
    config['cut'] = range(10,50)
    config["dim_feats"] = {'xyvp':[6,2], 'xyz':[4,3], 'xy':[3,2], 'xyp':[4,2], 'vp':[4,2], 'vpt':[5,2]}
    config['type_feats'] = 'vp'
    config['f'] = '100f'
    config['name'] = 'laneGCN'
    config['train_split'] = '/home/avt/prediction/Waymo/data_processed/' + config['type_feats'] + '/train_' + config['f'] 
    config['val_split'] = '/home/avt/prediction/Waymo/data_processed/' + config['type_feats'] + '/val_' + config['f']
    config['dd'] = date.today().strftime('%m%d')

    

    net = GreatNet(config)
    checkpoint = torch.load('/home/avt/prediction/Waymo/working/weights/laneGCN_vp_100f0720.pth') 
    net.load_state_dict(checkpoint)
    net.cuda()


    loss_f = Loss(config)
    loss_f.cuda()

    post = Postprocess(config)

    optimizer = optim.Adam(net.parameters(), lr = config['lr'])
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.08)

    batch_size = 4
    dataset_train = W_Dataset(config['train_split'])
    train_loader = DataLoader(dataset_train, 
                           batch_size = batch_size ,
                           collate_fn = collate_fn, 
                           shuffle = True, 
                           drop_last=True)
    
    dataset_val = W_Dataset(config['val_split'])
    val_loader = DataLoader(dataset_val, 
                           batch_size = batch_size ,
                           collate_fn = collate_fn, 
                           shuffle = True, 
                           drop_last=True)
    

    num_epochs = config['num_epochs']
    
    log_file = 'train.log'
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=log_file, level=logging.INFO, format=log_format)

    msg = 'Training info: ' + config['name'] + ',' + config['type_feats'] + ',' + config['f']
    print(msg)
    logging.info(msg)

    Tfde_a = 100
    for epoch in range(num_epochs):
        train1(net,train_loader,loss_f,optimizer,epoch,num_epochs,post)
        #scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            Tfde_a = val1(net,val_loader,loss_f,epoch,num_epochs, post, Tfde_a, config)


if __name__ == "__main__":
    main()