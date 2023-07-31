# with anchors

import sys
sys.path.append('/home/avt/prediction/Waymo/working/')

from math import gcd
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Dict, List, Tuple, Union
from utils import to_long, gpu
import numpy as np
from scipy import sparse
from layers import Linear,LinearRes,AttDest,actor_gather,graph_gather
from blocks import ActorNet,MapNet,A2A,M2M,M2A,A2M,Anchor


class PredNet(nn.Module):
    def __init__(self, config):
        super(PredNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        pred = []
        for i in range(config["num_mods"]):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2 * config["num_preds"]),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest(n_actor)
        self.cls = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng), nn.Linear(n_actor, 1)
        )

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Dict[str, List[Tensor]]:
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        n_c = self.config['dim_feats'][self.config['type_feats']][1]
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, n_c)
            reg[idcs] = reg[idcs] + ctrs[:,:,:,:2]

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs)
        cls = self.cls(feats).view(-1, self.config["num_mods"])

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"] = [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            out["cls"].append(cls[idcs])
            out["reg"].append(reg[idcs])
        return out



class GreatNet(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()

        self.config = config

        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)

        self.a2m = A2M(config)
        self.m2m = M2M(config)
        self.m2a = M2A(config)
        self.a2a = A2A(config)       
        self.anchor_net = Anchor(config)
       
        self.pred_net = PredNet(config)
    
    def forward(self, data: Dict) -> Tensor:

        num_acrs = len(self.config['acrs'])

        actors, actor_idcs = actor_gather(data["feats"])
        actor_ctrs = [torch.stack(i,0) for i in data["ctrs"]]

        actors = gpu(actors)
        actor_idcs = gpu(actor_idcs)
        actor_ctrs = gpu(actor_ctrs)

        actors = self.actor_net(actors) # (A, 128)

        #------------------------------------------------------------#

        graph = to_long(data['graph'])
        graph = graph_gather(graph)
        graph = gpu(graph)
        nodes, node_idcs, node_ctrs = self.map_net(graph) # (B, 128)

        #------------------------------------------------------------#
        
        ctrs_ =  actor_ctrs
        # nodes = self.a2m(nodes, graph, actors, actor_idcs, ctrs_)
        # nodes = self.m2m(nodes, graph)

        acrs_ = []
        for i in range(num_acrs + 1):
            actors = self.m2a(actors, actor_idcs, ctrs_, nodes, node_idcs, node_ctrs)
            actors = self.a2a(actors, actor_idcs, ctrs_) #(A,128)

            if i < num_acrs :
                acr_out, new_ctrs = self.anchor_net(actors, actor_idcs, ctrs_)
                ctrs_ = new_ctrs
                acrs_.append(acr_out)

        out = self.pred_net(actors, actor_idcs, actor_ctrs)

        for i in range(len(acrs_)):
            out['a_reg' + str(i)] = acrs_[i]['reg']
            out['a_cls' + str(i)] = acrs_[i]['cls']

        rot, orig = gpu(data['rot']), gpu(data['orig'])

        keys = [key for key in out.keys() if 'a_reg' in key]
        # to_global
        for i in range(len(out['reg'])):
            out['reg'][i] = torch.matmul(out['reg'][i], rot[i]) + orig[i][:2].view(1, 1, 1, -1)
            
            for key in keys:
                out[key][i] = torch.matmul(out[key][i], rot[i]) + orig[i][:2].view(1, 1, -1)

        return out
