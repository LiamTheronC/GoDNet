import os
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/avt/prediction/Waymo/working/')
from utils import pre_gather, gather

config = dict()
config["metrics_preds"] = [30,50,80]
config['num_mods'] = 6


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


def get_minFDE(post_out,data,metrics_preds,num_mods):
    #num_preds = np.array([30, 50, 80])
    minFDE = []
    f_idcs = []
    for j in range(len(metrics_preds)):
        reg,gt_preds,_,last_idcs, row_idcs = get_lastIdcs(post_out, metrics_preds[j],data)

        dist_6m = []
        for i in range(num_mods):

            rr = reg[row_idcs,i,last_idcs]
            gg = gt_preds[row_idcs,last_idcs].cuda()
            dist = torch.sqrt(((rr - gg)**2).sum(1))
            dist_6m.append(torch.tensor(dist).view(-1,1))

        zz = torch.cat(dist_6m,1)
        min_dist, min_idcs = zz.min(1)
        fde = min_dist.mean().item()

        minFDE.append(fde)
        f_idcs.append(min_idcs)
    
    mean = torch.tensor(minFDE).mean().item()
    minFDE.append(mean)

    return minFDE, f_idcs


def get_minADE(post_out,data,num_preds,num_mods):
    #num_preds = np.array([30, 50, 80])
    minADE = []
    for j in range(len(num_preds)):
        reg,gt_preds,has_preds,_,_ = get_lastIdcs(post_out, num_preds[j], data)

        dist_6m = []
        for i in range(num_mods):
            
            dist = []
            for j in range(len(reg)):
                rr = reg[j][i]
                gg = gt_preds[j].cuda()
                hh = has_preds[j].cuda()
                dd = torch.sqrt(((rr[hh] - gg[hh])**2).sum(1))
                dist.append(dd.mean().item())
            
            dist_6m.append(torch.tensor(dist).view(-1,1))

        zz = torch.cat(dist_6m,1)
        min_dist, min_idcs = zz.min(1)
        ade = min_dist.mean().item()
        minADE.append(ade)

    mean = torch.tensor(minADE).mean().item()
    minADE.append(mean)
    
    return minADE


class Postprocess():
    def __init__(self,config) -> None:
        self.config = config
    
    def append(self,metrics,loss,post_out,data):
        

        minFDE,f_idcs = get_minFDE(post_out,
                            data,
                            config["metrics_preds"],
                            config["num_mods"])
        
        minADE = get_minADE(post_out,
                            data,
                            config["metrics_preds"],
                            config["num_mods"])
        
        m = dict()
        
        m['loss'] = loss
        m['fde'] = minFDE[2]
        m['ade'] = minADE[2]
        m['f_idcs'] = f_idcs

        for key in m.keys():
            if key in metrics.keys():
                metrics[key].append(m[key])
            else:
                metrics[key] = [m[key]]
        
       
        return metrics


    def display(self, metrics, dt, epoch, num_epochs, mode="Train"):

        out= []
        for key in metrics.keys():
            if key in ['loss','fde','ade']:
                out.append(sum(metrics[key])/len(metrics[key]))
    
        if mode == 'Train':
            msg1 = ' --- (' + mode + '), Epoch [{}/{}], Time:{:.1f} ---'.format(epoch+1, num_epochs, dt)
        elif mode == 'Validation':
            msg1 = ' ***(' + mode + '), Epoch [{}/{}], Time:{:.1f} ***'.format(epoch+1, num_epochs, dt)
        
        msg2 = 'loss:{:.2f} --- fde:{:.2f} --- ade:{:.2f}'.format(out[0],out[1],out[2])

        print(msg1)
        print(msg2)
        logging.info(msg1)
        logging.info(msg2)

        return msg2
    

    def plot(self, metrics, data, outputs, msg, key = 1):

        f_idcs = metrics['f_idcs'][0][2]
        row_idcs = torch.tensor(range(len(f_idcs)))

        gt_pred, has_pred = pre_gather(data['gt_preds']), pre_gather(data['has_preds'])
        
        # prediction
        if key == 1:# plot only the minFDE of each object
            regs = outputs['reg'][0][row_idcs,f_idcs].cpu().detach()
            for j,reg in enumerate(regs):
                plt.plot(reg[has_pred[j]].T[0],reg[has_pred[j]].T[1],color='green',linewidth=0.8)
        elif key == 6:# plot all the six outputs of each object
            regs = outputs['reg'][0]
            for i in range(config["num_mods"]):
                reg = regs[:,i].cpu().detach()
                for j,z in enumerate(reg):
                    plt.plot(z[has_pred[j]].T[0],z[has_pred[j]].T[1],color='green',linewidth=0.8)
        
        
        # ground truth
        for i, gt in enumerate(gt_pred):
            tmp = gt[has_pred[i]]
            if len(tmp)>0:
                plt.plot(tmp.T[0],tmp.T[1],color='red', linewidth = 1.0, linestyle='--')
                plt.scatter(tmp.T[0][0],tmp.T[1][0],s=30, color ='red')
        
        # graph
        rot = data['rot'][0]
        orig = data['orig'][0]
        ctrs = data['graph'][0]['ctrs'][:, :2]
        ctrs = torch.matmul(ctrs, rot) + orig[:2]
        plt.scatter(ctrs.T[0] ,ctrs.T[1], c = 'black',s = 0.05)


        # trajectory history 
        trajs = data['trajs_xyz'][0]
        masks = data['valid_masks'][0]
        for i,traj in enumerate(trajs):
            traj = traj[:12][masks[i][:12]]
            plt.plot(traj.T[0],traj.T[1],color='blue',linewidth=1.0, linestyle='--')

        
        plt.gca().set_aspect('equal')
        plt.title(msg)
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')
        plt.show()

