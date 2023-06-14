# M1 training


import os
import numpy as np
from fractions import gcd
from numbers import Number
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from net_M1 import GreatNet, Loss, pre_gather, PostProcess
import torch.optim as optim
import random
from utils import collate_fn
import logging
from memory_profiler import profile
import gc
import time


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



def train(net,train_loader,loss_f,optimizer,epoch,num_epochs, postprocess):

    net.train()
    metrics = dict()
    metrics['loss'], metrics['fde'], metrics['ade'] = [], [], []

    start_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        #print(feat.shape,gt_pred.shape,has_pred.shape,ctrs.shape)

        outputs = net(data)
        loss = loss_f(outputs, data)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        post_out = postprocess(outputs, data)
        postprocess.append(metrics, post_out, loss)
        
    dt = time.time() - start_time
    postprocess.display(metrics, dt, epoch, num_epochs, str='Train')

    
    
def val(net,val_loader,loss_f,epoch,num_epochs, postprocess):

    net.eval()
    metrics = dict()
    metrics['loss'], metrics['fde'], metrics['ade'] = [], [], []

    with torch.no_grad():
        start_time = time.time()
        for batch_idx, data in enumerate(val_loader):
            outputs = net(data)
            loss = loss_f(outputs, data)
            
            post_out = postprocess(outputs, data)
            postprocess.append(metrics, post_out, loss)

    dt = time.time() - start_time
    postprocess.display(metrics, dt, epoch, num_epochs, str='Validation')


def train1(net,train_loader,loss_f,optimizer,epoch,num_epochs):
    # output is trajectory
    net.train()
    loss_t = []
    for batch_idx, data in enumerate(train_loader):
        #print(feat.shape,gt_pred.shape,has_pred.shape,ctrs.shape)

        outputs = net(data)
        outputs = outputs.view(outputs.size(0),-1,2)

        ctrs = pre_gather(data['ctrs'])
        ctrs = ctrs[:,:2].unsqueeze(1).cuda()

        outputs += ctrs

        has_pred = pre_gather(data['has_preds']).cuda()
        gt_pred = pre_gather(data['gt2_preds']).float().cuda()

        loss = loss_f(outputs, gt_pred, has_pred)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_t.append(loss)

    mean_loss = sum(loss_t)/len(loss_t)
    msg = 'Epoch [{}/{}], Train_Loss: {:.4f}'.format(epoch+1, num_epochs, mean_loss)
    print(msg)
    logging.info(msg)


def val1(net,val_loader,loss_f,epoch,num_epochs):
    net.eval()
    loss_v = []
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            outputs = net(data)
            outputs = outputs.view(outputs.size(0),-1,2)

            ctrs = pre_gather(data['ctrs'])
            ctrs = ctrs[:,:2].unsqueeze(1).cuda()

            outputs += ctrs

            has_pred = pre_gather(data['has_preds']).cuda()
            gt_pred = pre_gather(data['gt2_preds']).float().cuda()

            loss = loss_f(outputs, gt_pred, has_pred)
            loss_v.append(loss)
    
    mean_loss = sum(loss_v)/len(loss_v)
    msg = 'Epoch [{}/{}], Val_Loss: {:.4f}'.format(epoch+1, num_epochs, mean_loss)
    print(msg)
    logging.info(msg)



@profile
def main():
    seed = 33

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = dict()
    config['n_actornet'] = 128
    config['num_epochs'] = 50
    config['lr'] = 1e-3
    config['train_split'] = '/home/avt/prediction/Waymo/data_processed/train1'
    config['val_split'] = '/home/avt/prediction/Waymo/data_processed/val1'
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
    config["cls_th"] = 2.0
    config["cls_ignore"] = 0.2
    config["mgn"] = 0.2
    config["cls_coef"] = 1.0
    config["reg_coef"] = 1.0

    net = GreatNet(config)
    net.cuda()

    loss_f = Loss()
    loss_f.cuda()

    postprocess = PostProcess(config)

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


    for epoch in range(num_epochs):
        train(net,train_loader,loss_f,optimizer,epoch,num_epochs,postprocess)
        if epoch * ((epoch + 1) % 10) == 0:
            val(net,val_loader,loss_f,epoch,num_epochs,postprocess)

    torch.save(net.state_dict(), 'model_Great_m1_weights_e300.pth')

if __name__ == "__main__":
    main()