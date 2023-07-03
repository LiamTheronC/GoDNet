import sys
sys.path.append('/home/avt/prediction/Waymo/working/')

import numpy as np
from fractions import gcd
from numbers import Number
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from memory_profiler import profile
from utils import to_long, gpu, pre_gather


class PredLoss(nn.Module):
    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out: Dict[str, List[Tensor]], data) -> Dict[str, Union[Tensor, int]]:
        cls, reg = out["cls"], out["reg"]
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
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
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



class MidLoss(nn.Module):
    def __init__(self, config) -> None:
        super(MidLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")
    
    def forward(self, out: Dict[str, List[Tensor]], data) -> Dict[str, Union[Tensor, int]]:

        mid_out = dict()
        mid_num = self.config['mid_num']
        num_mods = self.config["num_mods"]

        mid = torch.cat(out['mid'],0)
        cls = torch.cat(out['cls'],0)
        has_mid = pre_gather(data['has_preds'])[:, mid_num - 1].cuda()
        gt_mid = pre_gather(data['gt_preds']).float()[:, mid_num - 1,:2].cuda()

        mid = mid[has_mid]
        cls = cls[has_mid]
        gt_mid = gt_mid[has_mid]

        dist = torch.sqrt(
            (
                (mid - gt_mid.unsqueeze(1))
                ** 2
            ).sum(2)
        )

        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mid_out['reg_loss'] = self.reg_loss(mid[row_idcs,min_idcs],gt_mid)
        mid_out['num_reg'] = has_mid.sum().item()

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        coef = self.config["cls_coef"]
        
        mid_out["cls_loss"] = coef * (self.config["mgn"] * mask.sum() - mgn[mask].sum())
        mid_out["num_cls"] = mask.sum().item()

        return mid_out




class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)
        self.mid_loss = MidLoss(config)

    def forward(self, out: Dict, data: Dict) -> Dict:
        loss_out = self.pred_loss(out, data)
        mid_out = self.mid_loss(out,data)

        loss_out["loss"] = loss_out["cls_loss"] / (
            loss_out["num_cls"] + 1e-10) + loss_out["reg_loss"] / (
            loss_out["num_reg"] + 1e-10) + mid_out['reg_loss'] / (
            mid_out['num_reg'] + 1e-10) + mid_out['cls_loss'] / (
            mid_out["num_cls"] + 1e-10
        )
       
        return loss_out