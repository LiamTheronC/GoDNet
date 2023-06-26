# M6 training

import os
import numpy as np
from fractions import gcd
from numbers import Number
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from model.laneGCN import GreatNet
from losses.loss import Loss
import torch.optim as optim
import random
from utils import collate_fn, pre_gather
import logging
from memory_profiler import profile
from metrics.metrics import Postprocess
import gc
import time
from datetime import date



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



# def train(net,train_loader,loss_f,optimizer,epoch,num_epochs):
#     net.train()
#     loss_t = []
#     for batch_idx, data in enumerate(train_loader):
        

#         outputs = net(data)
#         outputs = outputs.view(outputs.size(0),-1,2)

#         ctrs = pre_gather(data['ctrs'])
#         ctrs = ctrs[:,:2].unsqueeze(1).cuda()

#         outputs = outputs.cumsum(dim=1) + ctrs

#         has_pred = pre_gather(data['has_preds']).cuda()
#         gt_pred = pre_gather(data['gt_preds']).float().cuda()

#         loss = loss_f(outputs, gt_pred, has_pred)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         loss_t.append(loss)

#     mean_loss = sum(loss_t)/len(loss_t)
#     msg = 'Epoch [{}/{}], Train_Loss: {:.4f}'.format(epoch+1, num_epochs, mean_loss)
#     print(msg)
#     logging.info(msg)


# def val(net,val_loader,loss_f,epoch,num_epochs):
#     net.eval()
#     loss_v = []
#     with torch.no_grad():
#         for batch_idx, data in enumerate(val_loader):
#             outputs = net(data)
#             outputs = outputs.view(outputs.size(0),-1,2)

#             ctrs = pre_gather(data['ctrs'])
#             ctrs = ctrs[:,:2].unsqueeze(1).cuda()

#             outputs = outputs.cumsum(dim=1) + ctrs

#             has_pred = pre_gather(data['has_preds']).cuda()
#             gt_pred = pre_gather(data['gt_preds']).float().cuda()

#             loss = loss_f(outputs, gt_pred, has_pred)
#             loss_v.append(loss)
    
#     mean_loss = sum(loss_v)/len(loss_v)
#     msg = 'Epoch [{}/{}], Val_Loss: {:.4f}'.format(epoch+1, num_epochs, mean_loss)
#     print(msg)
#     logging.info(msg)



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
        loss_out['loss'].backward()
        optimizer.step()

        post.append(metrics,loss_out['loss'].item(),outputs,data)

    dt = time.time() - start_time
    post.display(metrics, dt, epoch, num_epochs, "Train")


def val1(net,val_loader,loss_f,epoch,num_epochs,post):
    net.eval()
    metrics = dict()
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):

            outputs = net(data)
            loss_out = loss_f(outputs,data)
            post.append(metrics,loss_out['loss'].item(),outputs,data)
    
    dt = time.time() - start_time
    post.display(metrics, dt, epoch, num_epochs, "Validation")


@profile
def main():
    seed = 33

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    current_date = date.today()
    dd = current_date.strftime('%m%d')

    config = dict()
    config['n_actornet'] = 128
    config['num_epochs'] = 150
    config['lr'] = 1e-3
    config["num_scales"] = 6
    config["n_map"] = 128
    config["n_actor"] = 128
    config["actor2map_dist"] = 7.0
    config["map2actor_dist"] = 6.0
    config["actor2actor_dist"] = 100.0
    config["num_mods"] = 6
    config["pred_size"] = 80
    config["pred_step"] = 1
    config["num_preds"] = config["pred_size"] // config["pred_step"]
    config["cls_th"] = 2.0 #5.0
    config["cls_ignore"] = 0.2
    config["mgn"] = 0.2
    config["cls_coef"] = 1.0
    config["reg_coef"] = 1.0
    config["metrics_preds"] = [30,50,80]
    config["dim_feats"] = {'xyvp':[6,2], 'xyz':[4,3], 'xy':[3,2], 'xyp':[4,2], 'vp':[4,2]}
    config['type_feats'] = 'vp'
    config['f'] = '1f'
    config['train_split'] = '/home/avt/prediction/Waymo/data_processed/' + config['type_feats'] + '/train_' + config['f'] 
    config['val_split'] = '/home/avt/prediction/Waymo/data_processed/' + config['type_feats'] + '/val_' + config['f']

    net = GreatNet(config)
    net.cuda()

    loss_f = Loss(config)
    loss_f.cuda()

    post = Postprocess(config)

    optimizer = optim.Adam(net.parameters(), lr = config['lr'])

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

    msg = 'Training info: ' + 'LaneGCN' + ',' + config['type_feats'] + ',' + config['f']
    print(msg)
    logging.info(msg)

    for epoch in range(num_epochs):
        train1(net,train_loader,loss_f,optimizer,epoch,num_epochs,post)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val1(net,val_loader,loss_f,epoch,num_epochs,post)
        torch.save(net.state_dict(), 'weights/laneGCN_'+ config['type_feats'] + '_' + config['f'] + dd +'.pth')


if __name__ == "__main__":
    main()