import os
import numpy as np
from fractions import gcd
from numbers import Number
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from net_M6 import GreatNet,Loss,pre_gather
import torch.optim as optim
import random
from memory_profiler import profile
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
    config['n_actornet'] = 128
    config['num_epochs'] = 50
    config['lr'] = 2e-3
    config['train_split'] = '/home/avt/prediction/Waymo/data_processed/train'
    config['val_split'] = '/home/avt/prediction/Waymo/data_processed/validation'
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

    config['model_weights'] = 'model_Great_m1_weights.pth'

    net = GreatNet(config)
    net.load_state_dict(torch.load(config['model_weights']))
    net.cuda()

    loss_f = Loss(config)
    loss_f.cuda()
    
    
    batch_size = 1
    # dataset_train = W_Dataset(config['train_split'])
    # train_loader = DataLoader(dataset_train, 
    #                        batch_size = batch_size,
    #                        collate_fn = collate_fn, 
    #                        shuffle = True, 
    #                        drop_last=True)
    
    dataset_val = W_Dataset(config['train_split'])
    val_loader = DataLoader(dataset_val, 
                           batch_size = batch_size ,
                           collate_fn = collate_fn, 
                           shuffle = True, 
                           drop_last=True)

    num_epochs = config['num_epochs']

    net.eval()
    with torch.no_grad():
        for epoch in range(num_epochs):
            for batch_idx, data in enumerate(val_loader):
                #print(feat.shape,gt_pred.shape,has_pred.shape,ctrs.shape)

                outputs = net(data)
                loss_out = loss_f(outputs,data)
                print(loss_out['loss'].item())
                
                has_pred = pre_gather(data['has_preds'])
                gt_pred = pre_gather(data['gt2_preds']).float()
                print(outputs.size(),has_pred.size(),gt_pred.size())

                
                gt_pred = gt_pred.cpu().detach()
                has_pred = has_pred.cpu().detach()
                # regs = outputs['reg']
                # tt = loss_out['reg'].cpu().detach()

                # for g in data['graph']:
                #      plt.scatter(g['ctrs'].T[0] ,g['ctrs'].T[1], c = 'black',s = 0.05)
                

                # for reg in regs:
                #     reg = reg.cpu().detach()
                #     reg = reg.view(-1,80,2)
                #     for z in reg:
                #         plt.plot(z.T[0],z.T[1],color='green',linewidth=0.8, linestyle='--')
                
                # for i, gt in enumerate(gt_pred):
                #     tmp = gt[has_pred[i]]
                #     t = tt[i][has_pred[i]]

                #     if len(tmp)>0:
                #         plt.plot(tmp.T[0],tmp.T[1],color='red')
                #         plt.scatter(tmp.T[0][0],tmp.T[1][0],s=30, color ='red')
                #         plt.plot(t.T[0],t.T[1],color='blue',linewidth=1, linestyle='--')

                # # for t in tt:
                # #     plt.plot(t.T[0],t.T[1],color='blue',linewidth=1, linestyle='--')


                
                # plt.gca().set_aspect('equal')
                # plt.show()

                for g in data['graph']:
                    plt.scatter(g['ctrs'].T[0] ,g['ctrs'].T[1], c = 'black',s = 0.05)

                for i, gt in enumerate(gt_pred):
                    tmp = gt[has_pred[i]]
                    out = outputs[i][has_pred[i]]

                    if len(tmp)>0:
                        plt.plot(tmp.T[0],tmp.T[1],color='red')
                        plt.scatter(tmp.T[0][0],tmp.T[1][0],s=30, color ='red')
                        plt.plot(out.T[0],out.T[1],color='blue', linewidth=1, linestyle='--')
                
                plt.gca().set_aspect('equal')
                plt.show()
                break
            break

if __name__ == "__main__":
    main()
    

