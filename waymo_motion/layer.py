import numpy as np
from fractions import gcd
from numbers import Number
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Linear(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=2, act=True):
        super(Linear, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.linear = nn.Linear(n_in, n_out, bias=False)
        
        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')
        
        self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.Sigmoid()
        self.act = act

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class MLP(nn.Module):
    def __init__(self) -> None:
        super(MLP,self).__init__()

        self.L1 = Linear(44,128)
        self.L2 = Linear(128,128)
        self.L3 = Linear(128,160)
        self.L4 = Linear(160,160,act=False)
    
    def forward(self,x: dict) -> Tensor:
        
        actors, actor_idcs = actor_gather(x['feats'])
        actors = actors.view(actors.size(0),44).to(device)
        out = self.L1(actors)
        out = self.L2(out)
        out = self.L3(out)
        out = self.L4(out)

        return out


class Loss(nn.Module):
    def __init__(self) -> None:
        super(Loss,self).__init__()
        self.MSE = nn.MSELoss()
    
    def forward(self, out, gt_preds, has_preds):

        prediction = out[has_preds]
        target = gt_preds[has_preds][:,:2]
        
        loss = self.MSE(prediction, target)
    
        return loss

def actor_gather(actors: List) -> Tuple[Tensor, List[Tensor]]:
    """
    actors is data['feat']
     
    """

    batch_size = len(actors)
    num_actors = [len(x) for x in actors]

    actors = [torch.stack(x).transpose(1, 2) for x in actors]
    actors = torch.cat(actors, 0)

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
        actor_idcs.append(idcs)
        count += num_actors[i]
    return actors, actor_idcs



def pre_gather(gts: List) -> Tensor:
    tmp = list()
    for g in gts:
        tmp += g
    
    tmp = torch.stack(tmp)

    return tmp
