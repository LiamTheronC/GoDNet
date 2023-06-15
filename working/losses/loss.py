# M6 net

import numpy as np
from fractions import gcd
from numbers import Number
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from utils import to_long, gpu
from memory_profiler import profile

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
        gt_preds = pre_gather(data['gt2_preds']).float()[:,:,:2].cuda()

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
        loss_out['reg'] = reg
        return loss_out


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)

    def forward(self, out: Dict, data: Dict) -> Dict:
        loss_out = self.pred_loss(out, data)
        loss_out["loss"] = loss_out["cls_loss"] / (
            loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out
    

def pre_gather(gts: List) -> Tensor:
    tmp = list()
    for g in gts:
        tmp += g
    
    tmp = torch.stack(tmp)

    return tmp

