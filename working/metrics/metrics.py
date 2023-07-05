import sys
sys.path.append('/home/avt/prediction/Waymo/working/')

import os
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from utils import pre_gather, gather

config = dict()
config["metrics_preds"] = [30,50,80]
config['num_mods'] = 6


# def get_lastIdcs(reg, gt_preds, has_preds, num_pred):

#     reg = torch.cat([x for x in reg], 0)
#     gt_preds = torch.cat([x for x in gt_preds], 0)
#     has_preds = torch.cat([x for x in has_preds], 0)


#     last = has_preds.float() + 0.1 * torch.arange(num_pred).float().to(
#                 has_preds.device
#             ) / float(num_pred)

#     max_last, last_idcs = last.max(1)
#     mask = max_last >1.0

#     reg = reg[mask]
#     gt_preds = gt_preds[mask][:,:,:2]
#     has_preds = has_preds[mask]
#     last_idcs = last_idcs[mask]

#     row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)

#     return reg, gt_preds, has_preds, last_idcs, row_idcs


def get_lastIdcs(reg, num_pred, gt_preds, has_preds):

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


def get_minFDE(reg,data,metrics_preds,num_mods, target = False):
    #num_preds = np.array([30, 50, 80])
    minFDE = []
    f_idcs = []

    gt_preds, has_preds = gather(data['gt_preds']), gather(data['has_preds'])
    
    r_t,g_t,h_t = get_target(reg, gt_preds, has_preds, data['target_indx_e'],target)

    for j in range(len(metrics_preds)):
        reg, gt_preds,_,last_idcs, row_idcs = get_lastIdcs(r_t, metrics_preds[j], g_t, h_t)
        dist_6m = []
        for i in range(num_mods):

            rr = reg[row_idcs,i,last_idcs]
            gg = gt_preds[row_idcs,last_idcs].cuda()
            dist = torch.sqrt(((rr - gg)**2).sum(1))
            dist_6m.append(dist.clone().detach().view(-1,1))

        zz = torch.cat(dist_6m,1)
        min_dist, min_idcs = zz.min(1)
        fde = min_dist.mean().item()

        minFDE.append(fde)
        f_idcs.append(min_idcs)
    
    mean = torch.tensor(minFDE).mean().item()
    minFDE.append(mean)

    return minFDE, f_idcs



# def get_minFDE(reg, data, num_preds, num_mods):
#     #num_preds = np.array([30, 50, 80])
#     minFDE = []
#     gt_preds, has_preds = gather(data['gt_preds']), gather(data['has_preds'])

#     indx_final = get_tarIndx(reg, data)

#     reg = torch.cat([x for x in reg], 0)
#     gt_preds = torch.cat([x for x in gt_preds], 0)
#     has_preds = torch.cat([x for x in has_preds], 0)



#     reg, gt_preds,_,last_idcs, row_idcs = get_lastIdcs(reg, gt_preds, has_preds, num_preds)
    
#     dist_6m = []

#     for i in range(num_mods):

#         rr = reg[row_idcs,i,last_idcs]
#         gg = gt_preds[row_idcs,last_idcs].cuda()
#         dist = torch.sqrt(((rr - gg)**2).sum(1))
#         dist_6m.append(dist.clone().detach().view(-1,1))

#     zz = torch.cat(dist_6m,1)
#     min_dist, min_idcs = zz.min(1)
#     fde = min_dist.mean().item()

#     minFDE.append(fde)
    
#     mean = torch.tensor(minFDE).mean().item()
#     minFDE.append(mean)

#     return minFDE


# def get_minADE(reg,data,metrics_preds,num_mods,target = False):
#     #num_preds = np.array([30, 50, 80])
#     minADE = []
#     gt_preds, has_preds = gather(data['gt_preds']), gather(data['has_preds'])
#     r_t,g_t,h_t = get_target(reg,gt_preds, has_preds,data['target_indx_e'],target)

#     for j in range(len(metrics_preds)):
#         reg,gt_preds,has_preds,_,_ = get_lastIdcs(r_t, metrics_preds[j], g_t, h_t)

#         dist_6m = []
#         for i in range(num_mods):
            
#             dist = []
#             for j in range(len(reg)):
#                 rr = reg[j][i]
#                 gg = gt_preds[j].cuda()
#                 hh = has_preds[j].cuda()
#                 dd = torch.sqrt(((rr[hh] - gg[hh])**2).sum(1))
#                 dist.append(dd.mean().item())
            
#             dist_6m.append(torch.tensor(dist).view(-1,1))

#         zz = torch.cat(dist_6m,1)
#         min_dist, min_idcs = zz.min(1)
#         ade = min_dist.mean().item()
#         minADE.append(ade)

#     mean = torch.tensor(minADE).mean().item()
#     minADE.append(mean)
    
#     return minADE


def get_minADE(reg,data,metrics_preds,num_mods,target = False):
    #num_preds = np.array([30, 50, 80])
    minADE = []
    gt_preds, has_preds = gather(data['gt_preds']), gather(data['has_preds'])
    r_t,g_t,h_t = get_target(reg,gt_preds, has_preds,data['target_indx_e'],target)

    for j in range(len(metrics_preds)):
        reg,gt_preds,has_preds,_,_ = get_lastIdcs(r_t, metrics_preds[j], g_t, h_t)

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
        

        minFDE,f_idcs = get_minFDE(post_out['reg'],
                            data,
                            config["metrics_preds"],
                            config["num_mods"])

        
        minADE = get_minADE(post_out['reg'],
                            data,
                            config["metrics_preds"],
                            config["num_mods"])
        
        minF,t_idcs = get_minFDE(post_out['reg'],
                            data,
                            config["metrics_preds"],
                            config["num_mods"],
                            True)
        
        minA = get_minADE(post_out['reg'],
                            data,
                            config["metrics_preds"],
                            config["num_mods"],
                            True)


        m = dict()
        
        m['loss'] = loss
        m['fde'] = minFDE[2]
        m['ade'] = minADE[2]
        m['Tfde'] = minF[2]
        m['Tade'] = minA[2]
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
            if key in ['loss','fde','ade','Tfde','Tade']:
                out.append(sum(metrics[key])/len(metrics[key]))
    
        if mode == 'Train':
            msg1 = ' --- (' + mode + '), Epoch [{}/{}], Time:{:.1f} ---'.format(epoch+1, num_epochs, dt)
        elif mode == 'Validation':
            msg1 = ' ***(' + mode + '), Epoch [{}/{}], Time:{:.1f} ***'.format(epoch+1, num_epochs, dt)
        
        msg2 = 'loss:{:.2f} -- fde:{:.2f} -- ade:{:.2f} -- Tfde:{:.2f} -- Tade:{:.2f}'.format(out[0],out[1],out[2],out[3],out[4])

        print(msg1)
        print(msg2)
        logging.info(msg1)
        logging.info(msg2)

        return msg2,out[0]
    

    def plot(self, metrics, data, outputs, msg, key = 1, target = False):

        f_idcs = metrics['f_idcs'][0][2]
        row_idcs = torch.tensor(range(len(f_idcs)))
        reg = outputs['reg']

        g_t, h_t = gather(data['gt_preds']), gather(data['has_preds'])
        r_t,g_t,h_t = get_target(reg,g_t, h_t,data['target_indx_e'], target)

        # the final sets for plot
        regs, gt_preds,has_preds,_,_ = get_lastIdcs(r_t, 80, g_t, h_t)
        
        # prediction
        if key == 1:# plot only the minFDE of each object
            regs = regs[row_idcs,f_idcs].cpu().detach()
            for j in range(len(regs)):
                line = regs[j][has_preds[j]]
                plt.plot(line.T[0],line.T[1],color='green',linewidth=0.8)

        elif key == 6:# plot all the six outputs of each object
            for i in range(config["num_mods"]):
                ri = regs[:,i].cpu().detach()
                for j in range(len(ri)):
                    line = ri[j][has_preds[j]]
                    plt.plot(line.T[0],line.T[1],color='green', linestyle='--', linewidth=0.8, alpha = 0.2)
            
            regs = regs[row_idcs,f_idcs].cpu().detach()
            for j in range(len(regs)):
                line = regs[j][has_preds[j]]
                plt.plot(line.T[0], line.T[1], color='green', linewidth=1.0)
        
        # ground truth
        for j in range(len(gt_preds)):
            line = gt_preds[j][has_preds[j]]
            if len(line) > 0:
                plt.plot(line.T[0],line.T[1],color='red', linewidth = 1.0, linestyle='--')
                plt.scatter(line.T[0][0],line.T[1][0],s=30, color ='red')
        
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


def get_target(reg, gt_preds, has_preds, indx,target):
    """
    return only the info of targets_to_predict if target is True

    Return: List[Tensors]

    """

    if target:
        r,g,h=[],[],[]
    
        for i in range(len(indx)):
            mask = torch.tensor(indx[i])
            r.append(reg[i][mask])
            g.append(gt_preds[i][mask])
            h.append(has_preds[i][mask])
        
        return r, g, h
    
    else:
        return reg, gt_preds, has_preds




# def get_tarIndx(reg, data):

#     indx = [torch.tensor(x) for x in data['target_indx_e']]
#     num = torch.tensor([len(x) for x in reg])
#     num = torch.cumsum(num,0)[:-1]

#     indx_cum = []
#     indx_cum.append(indx[0])
    
#     for ii in range(len(num)):
#         indx_cum.append(indx[ii+1] + num[ii].item())

#     indx_final = torch.cat(indx_cum,0)

#     return indx_final

def panorama(data,path):

    rot = data['rot']
    orig = data['orig']
    ctrs = data['graph']['ctrs'][:, :2]
    ctrs = np.matmul(ctrs, rot) + orig[:2]
    plt.scatter(ctrs.T[0] ,ctrs.T[1], c = 'black',s = 0.05)

    lanes = data['road_info']['lane']
    if data['road_info']['dynamic_map']:
        d_lanes = data['road_info']['dynamic_map'].keys()
        for l in d_lanes:
            line = lanes[l]['polyline']
            plt.plot(line.T[0],line.T[1],color = 'red', linewidth = 0.5)

    if 'crosswalk' in data['road_info'].keys():
        crosswalk = data['road_info']['crosswalk']['polygon']
        for c in crosswalk:
            c = np.concatenate((c,c[0:1]),0)
            plt.plot(c.T[0],c.T[1],color = 'green', linewidth = 0.5)

    if 'speedBump' in data['road_info'].keys():
        speedBump = data['road_info']['speedBump']['polygon']
        for s in speedBump:
            s = np.concatenate((s,s[0:1]),0)
            plt.plot(s.T[0],s.T[1],color = 'salmon', linewidth = 2)
    
    if 'driveway' in data['road_info'].keys():
        driveway = data['road_info']['driveway']['polygon']
        for d in driveway:
            d = np.concatenate((d,d[0:1]),0)
            plt.plot(d.T[0],d.T[1],color = 'brown', linewidth = 0.5)

    if 'roadLine' in data['road_info'].keys():
        road_line = data['road_info']['roadLine']['polyline']
        road_type = data['road_info']['roadLine']['type']

        for i in range(len(road_line)):
            l = road_line[i]
            if 'WHITE' in road_type[i]:
                c = 'skyblue'
            elif 'YELLOW' in road_type[i]:
                c = 'orange'
            else:
                c = 'black'
            
            if 'SOLID' in road_type[i]:
                s = '-'
            elif 'BROKEN' in road_type[i]:
                s = '--'
            else:
                s = '-.'
            
            plt.plot(l.T[0],l.T[1],color = c, linestyle = s, linewidth = 1)
    

    if 'roadEdge' in data['road_info'].keys():
        edge_line = data['road_info']['roadEdge']['polyline']
        edge_type = data['road_info']['roadEdge']['type']

        for i in range(len(edge_line)):
            e = edge_line[i]
            if 'MEDIAN' in edge_type[i]:
                c = 'black'
            elif 'BOUNDARY' in edge_type[i]:
                c = 'black'
            else:
                c = 'black'
            plt.plot(e.T[0],e.T[1],color = c, linestyle = '--', linewidth = 0.5)

  
    plt.gca().set_aspect('equal')
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.savefig(path, dpi=2000)
