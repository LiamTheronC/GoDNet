import sys
sys.path.append('/home/avt/prediction/Waymo/working/')

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Dict, List, Union
from utils import pre_gather


class PredLoss(nn.Module):
    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out: Dict[str, List[Tensor]], data) -> Dict[str, Union[Tensor, int]]:
        cls, reg = out["cls"], out["reg"]

        num = torch.tensor([len(x) for x in cls])

        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)
        has_preds = pre_gather(data['has_preds']).cuda()
        gt_preds = pre_gather(data['gt_preds']).float()[:,:,:2].cuda()

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0

        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"]
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        
        l = self.config['cut'] # range(10,50)

        # ade, find the idcs(out of 6) with the minimal ade in cut, for each agent -> min_idcs
        dist_= []
        for i in range(reg):
            rr = reg[i].transpose(0,1)[l] #(6,80,2) -> (80,6,2)
            gg = gt_preds[i].unsqueeze(1)[l] #(80,2) -> (80,1,2)
            hh = has_preds[i][l] #(80,)

            dd = torch.sqrt(((rr[hh] - gg[hh])**2).sum(2)).mean(0).unsqueeze(0) # (N,6) -> (6,) -> (1,6)
            dist_.append(dd)
        
        dist_ = torch.concatenate(dist_,0)
        a_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)
        
        # fde, find the fde corresponding to idcs, for each agent -> min_dist
        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                        (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist = dist[row_idcs, min_idcs]

        # loss calculation given (min_idcs, min_dist)
        
        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (a_dist < self.config["cls_th_2"]).view(-1, 1)
        mask1 = abs(dist - min_dist.view(-1, 1)) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        coef = self.config["cls_coef"]
        loss_out["cls_loss"] += coef * (
            self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_idcs, min_idcs]
        coef = self.config["reg_coef"]
        loss_out["reg_loss"] += coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()

        return loss_out



class AcrLoss(nn.Module):
    def __init__(self, config) -> None:
        super(AcrLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")
    
    def forward(self, out: Dict[str, List[Tensor]], data) -> Dict[str, Union[Tensor, int]]:

        acrs = self.config['acrs']
        acr_out = dict()

        for i in range(len(acrs)):

            acr_num = acrs[i]

            acr = torch.cat(out['a_reg' + str(i)],0)
            cls = torch.cat(out['a_cls' + str(i)],0)
            has_acr = pre_gather(data['has_preds'])[:, acr_num - 1].cuda()
            gt_acr = pre_gather(data['gt_preds']).float()[:, acr_num - 1,:2].cuda()

            acr = acr[has_acr]
            cls = cls[has_acr]
            gt_acr = gt_acr[has_acr]

            dist = torch.sqrt(
                (
                    (acr - gt_acr.unsqueeze(1))
                    ** 2
                ).sum(2)
            )

            min_dist, min_idcs = dist.min(1)
            row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

            acr_out['reg_loss' + str(i)] = self.reg_loss(acr[row_idcs,min_idcs], gt_acr)
            acr_out['num_reg' + str(i)] = has_acr.sum().item()

            mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
            mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
            mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
            mgn = mgn[mask0 * mask1]
            mask = mgn < self.config["mgn"]
            coef = self.config["cls_coef"]
            
            acr_out['cls_loss' + str(i)] = coef * (self.config["mgn"] * mask.sum() - mgn[mask].sum())
            acr_out['num_cls' + str(i)] = mask.sum().item()

        return acr_out



class Loss(nn.Module):
    # integrate mid goal loss
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)
        self.acr_loss = AcrLoss(config)

    def forward(self, out: Dict, data: Dict) -> Dict:

        acrs = self.config['acrs']

        loss_out = self.pred_loss(out, data)
        acr_out = self.acr_loss(out, data)

        loss = loss_out['cls_loss'] / (
            loss_out['num_cls'] + 1e-10) + loss_out['reg_loss'] / (
            loss_out['num_reg'] + 1e-10)
        
        for i in range(len(acrs)):
            loss += acr_out['cls_loss' + str(i)] / (
            acr_out['num_cls' + str(i)] + 1e-10) + acr_out['reg_loss' + str(i)] / (
            acr_out['num_reg' + str(i)] + 1e-10)
    
        return loss
    