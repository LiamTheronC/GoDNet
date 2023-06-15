import numpy as np
import torch

config = dict()
config["num_preds"] = [30,50,80]

def fx():
    return 0


def get_minFDE(post_out,num_preds):
   
    minFDE = []
    for j in range(len(num_preds)):
        reg, gt_preds, row_idcs, last_idcs = fx(num_preds[j])

        dist_6m = []
        for i in range(config["num_mods"]):

            rr = reg[row_idcs,i,last_idcs]
            gg = gt_preds[row_idcs,last_idcs].cuda()
            dist = torch.sqrt(((rr - gg)**2).sum(1))
            dist_6m.append(torch.tensor(dist).view(-1,1))

        zz = torch.cat(dist_6m,1)
        min_dist, min_idcs = zz.min(1)
        fde = min_dist.mean().item()

        minFDE.append(fde)
    
    mean = torch.tensor(minFDE).mean().item()
    minFDE.append(mean)

    return minFDE