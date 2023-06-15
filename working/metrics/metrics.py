import numpy as np
import torch

config = dict()
config["num_preds"] = [30,50,80]


def get_lastIdcs(out,num_pred,data):
    cls, reg = out["cls"], out["reg"]
    gt_preds, has_preds = gather(data['gt_preds']), gather(data['has_preds'])


    reg = torch.cat([x for x in reg], 0)
    gt_preds = torch.cat([x for x in gt_preds], 0)
    has_preds = torch.cat([x for x in has_preds], 0)

    reg = reg[:,:,:num_pred,:]
    gt_preds = gt_preds[:,:num_pred,:]
    has_preds = has_preds[:,:num_pred]

    last = has_preds.float() + 0.1 * torch.arange(num_pred).float().to(
                has_preds.device
            ) / float(num_pred)

    max_last, last_idcs = last.max(1)
    mask = max_last >1.0

    reg = reg[mask]
    gt_preds = gt_preds[mask][:,:,:2]
    has_preds = has_preds[mask]
    last_idcs = last_idcs[mask]

    row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)

    return reg, gt_preds, has_preds, last_idcs, row_idcs


def get_minFDE(post_out,num_preds,data):
    #num_preds = np.array([30, 50, 80])
    minFDE = []
    for j in range(len(num_preds)):
        reg,gt_preds,_,last_idcs, row_idcs = get_lastIdcs(post_out, num_preds[j],data)

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


def get_minADE(post_out,num_preds,data):

    return 0





def gather(gts) -> list:

    tmp = list()
    for i,g in enumerate(gts):
        zz = torch.stack(g, dim=0)
        tmp.append(zz)
    
    return tmp