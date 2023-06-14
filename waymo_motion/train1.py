import os
import numpy as np
from fractions import gcd
from numbers import Number
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from layer2 import GreatNet,Loss,pre_gather
import torch.optim as optim
import random
from memory_profiler import profile
from utils import collate_fn
import logging



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

def train(net,train_loader,device,loss_f,optimizer,epoch,num_epochs):
    net.train()
    loss_t = []
    for batch_idx, data in enumerate(train_loader):
        #print(feat.shape,gt_pred.shape,has_pred.shape,ctrs.shape)

        outputs = net(data)
        outputs = outputs.view(outputs.size(0),-1,2)

        ctrs = pre_gather(data['ctrs'])
        ctrs = ctrs[:,:2].unsqueeze(1).to(device)

        outputs = outputs.cumsum(dim=1) + ctrs

        has_pred = pre_gather(data['has_preds']).to(device)
        gt_pred = pre_gather(data['gt2_preds']).float().to(device)

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


def val(net,val_loader,device,loss_f,epoch,num_epochs):
    net.eval()
    loss_v = []
    for batch_idx, data in enumerate(val_loader):
        #print(feat.shape,gt_pred.shape,has_pred.shape,ctrs.shape)

        outputs = net(data)
        outputs = outputs.view(outputs.size(0),-1,2)

        ctrs = pre_gather(data['ctrs'])
        ctrs = ctrs[:,:2].unsqueeze(1).to(device)

        outputs = outputs.cumsum(dim=1) + ctrs

        has_pred = pre_gather(data['has_preds']).to(device)
        gt_pred = pre_gather(data['gt2_preds']).float().to(device)

        loss = loss_f(outputs, gt_pred, has_pred)
        loss_v.append(loss)
    
    mean_loss = sum(loss_v)/len(loss_v)
    msg = 'Epoch [{}/{}], Val_Loss: {:.4f}'.format(epoch+1, num_epochs, mean_loss)
    print(msg)
    logging.info(msg)
    return mean_loss


#@profile
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
    config['train_split'] = '/home/avt/prediction/Waymo/data_processed/train'
    config['val_split'] = '/home/avt/prediction/Waymo/data_processed/validation'


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net = GreatNet(config)
    net.to(device)

    loss_f = Loss()
    loss_f.to(device)

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
        train(net,train_loader,device,loss_f,optimizer,epoch,num_epochs)
        val(net,val_loader,device,loss_f,epoch,num_epochs)

    torch.save(net.state_dict(), 'model_Great_weights.pth')

if __name__ == "__main__":
    main()
    

