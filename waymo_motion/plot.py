import os
import numpy as np
from fractions import gcd
from numbers import Number
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from layer import MLP,Loss,pre_gather
import torch.optim as optim
from utils import collate_fn
import matplotlib.pyplot as plt


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


def main():
    config = dict()
    config['num_epochs'] = 100
    config['lr'] = 1e-3
    config['train_processed'] = '/home/avt/prediction/Waymo/data_processed/train/train_processed_5.pt'
    config['val_split'] = '/home/avt/prediction/Waymo/data_processed/validation'

    #device = "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    net = MLP()
    net.load_state_dict(torch.load('model_MLP_weights.pth'))
    net.to(device)

    loss_f = Loss()
    loss_f.to(device)
    
    
    batch_size = 4
    # dataset_train = W_Dataset(config['train_split'])
    # train_loader = DataLoader(dataset_train, 
    #                        batch_size = batch_size,
    #                        collate_fn = collate_fn, 
    #                        shuffle = True, 
    #                        drop_last=True)
    
    dataset_val = W_Dataset(config['val_split'])
    val_loader = DataLoader(dataset_val, 
                           batch_size = batch_size ,
                           collate_fn = collate_fn, 
                           shuffle = True, 
                           drop_last=True)

    num_epochs = config['num_epochs']
    

    for epoch in range(num_epochs):
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
            print(loss)

            gt_pred = gt_pred.cpu().detach()
            has_pred = has_pred.cpu().detach()
            outputs = outputs.cpu().detach()

            for i, gt in enumerate(gt_pred):
                tmp = gt[has_pred[i]]
                out = outputs[i][has_pred[i]]

                if len(tmp)>0:
                    plt.plot(tmp.T[0],tmp.T[1],color='red')
                    plt.scatter(tmp.T[0][0],tmp.T[1][0],s=30, color ='red')
                    plt.plot(out.T[0],out.T[1],color='blue')
            
            plt.gca().set_aspect('equal')
            plt.show()
            break
        break  

if __name__ == "__main__":
    main()
    

