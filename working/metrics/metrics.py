import sys
sys.path.append('/home/avt/prediction/Waymo/working/')

import os
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from utils import pre_gather, gather


def get_lastIdcs(reg, gt_preds, has_preds, num_pred):

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


def get_minFDE(reg, data, num_preds = 80, num_mods = 6, target = False):

    gt_preds, has_preds = gather(data['gt_preds']), gather(data['has_preds'])
    indx_final = get_tarIndx(reg, data)

    reg = torch.cat([x for x in reg], 0)
    gt_preds = torch.cat([x for x in gt_preds], 0)
    has_preds = torch.cat([x for x in has_preds], 0)

    if target:
        reg = reg[indx_final]
        gt_preds = gt_preds[indx_final]
        has_preds = has_preds[indx_final]

    reg, gt_preds,_,last_idcs, row_idcs = get_lastIdcs(reg, gt_preds, has_preds, num_preds)
    
    dist_6m = []

    for i in range(num_mods):

        rr = reg[row_idcs,i,last_idcs]
        gg = gt_preds[row_idcs,last_idcs].cuda()
        dist = torch.sqrt(((rr - gg)**2).sum(1))
        dist_6m.append(dist.clone().detach().view(-1,1))

    zz = torch.cat(dist_6m,1)
    min_dist, min_idcs = zz.min(1)

    fde = min_dist.mean().item()

    # if target:
    #     print('reg', reg[row_idcs, min_idcs, last_idcs])
    #     print(gt_preds[row_idcs, last_idcs])

    return fde, min_idcs


def get_minADE(reg, data, num_preds = 80, num_mods = 6, target = False):
   
    gt_preds, has_preds = gather(data['gt_preds']), gather(data['has_preds'])

    indx_final = get_tarIndx(reg, data)

    reg = torch.cat([x for x in reg], 0)
    gt_preds = torch.cat([x for x in gt_preds], 0)
    has_preds = torch.cat([x for x in has_preds], 0)

    if target:
        reg = reg[indx_final]
        gt_preds = gt_preds[indx_final]
        has_preds = has_preds[indx_final]

    reg, gt_preds,has_preds,_,_ = get_lastIdcs(reg, gt_preds, has_preds, num_preds)

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

    return ade



class Postprocess():
    def __init__(self,config) -> None:
        self.config = config
    
    def append(self, metrics, loss, outputs, data):
        

        fde, min_idcs = get_minFDE(outputs['reg'],
                            data,
                            self.config["num_preds"],
                            self.config["num_mods"])

        
        ade = get_minADE(outputs['reg'],
                            data,
                            self.config["num_preds"],
                            self.config["num_mods"])
        
        fde_t,_ = get_minFDE(outputs['reg'],
                            data,
                            self.config["num_preds"],
                            self.config["num_mods"],
                            True)

        
        ade_t = get_minADE(outputs['reg'],
                            data,
                            self.config["num_preds"],
                            self.config["num_mods"],
                            True)


        m = dict()
        
        m['loss'] = loss
        m['fde'] = fde
        m['ade'] = ade
        m['Tfde'] = fde_t
        m['Tade'] = ade_t
        m['f_idcs'] = min_idcs

        for key in m.keys():
            if key in metrics.keys():
                metrics[key].append(m[key])
            else:
                metrics[key] = [m[key]]
        
       
        return metrics


    def display(self, metrics, dt, epoch, num_epochs, mode = "Train"):

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

        return msg2,out[3]
    

    def plot(self, metrics, data, outputs, msg, key = 1, all = True):

        f_idcs = metrics['f_idcs'][0]
        
        row_idcs = torch.tensor(range(len(f_idcs)))
        reg = outputs['reg']
        gt_preds = gather(data['gt_preds'])
        has_preds = gather(data['has_preds'])
        engage_ids = data['engage_id'][0]
        target_ids = data['target_id'][0]

        reg = torch.cat([x for x in reg], 0)
        gt_preds = torch.cat([x for x in gt_preds], 0)
        has_preds = torch.cat([x for x in has_preds], 0)
        engage_ids = torch.tensor(engage_ids)
        target_ids = torch.tensor(target_ids)


        num_pred = self.config['num_preds']
        last = has_preds.float() + 0.1 * torch.arange(num_pred).float().to(
                has_preds.device
            ) / float(num_pred)

        max_last, last_idcs = last.max(1)
        mask = max_last >1.0

        regs = reg[mask]
        gt_preds = gt_preds[mask][:,:,:2]
        has_preds = has_preds[mask]
        engage_ids = engage_ids[mask]

        # prediction
        if key == 1:# plot only the minFDE of each object
            regs = regs[row_idcs,f_idcs].cpu().detach()
            for j in range(len(regs)):
                line = regs[j][has_preds[j]]
                if engage_ids[j] in target_ids:
                    plt.plot(line.T[0],line.T[1],color='black',linewidth=0.8)
                elif all:
                    plt.plot(line.T[0],line.T[1],color='green',linewidth=0.8)

        elif key == 6:# plot all the six outputs of each object
            for i in range(self.config["num_mods"]):
                ri = regs[:,i].cpu().detach()
                for j in range(len(ri)):
                    line = ri[j][has_preds[j]]
                    if engage_ids[j] in target_ids:
                        plt.plot(line.T[0],line.T[1],color='green', linewidth=0.8, alpha = 0.2)
                    elif all:
                        plt.plot(line.T[0],line.T[1],color='green', linewidth=0.8, alpha = 0.2)

            regs = regs[row_idcs,f_idcs].cpu().detach()
            for j in range(len(regs)):
                line = regs[j][has_preds[j]]
                if engage_ids[j] in target_ids:
                    plt.plot(line.T[0],line.T[1],color='black',linewidth=0.8)
                elif all:
                    plt.plot(line.T[0],line.T[1],color='green',linewidth=0.8)
        
        # ground truth
        for j in range(len(gt_preds)):
            line = gt_preds[j][has_preds[j]]
            if len(line) > 0:
                if engage_ids[j] in target_ids:
                    plt.plot(line.T[0],line.T[1],color='red', linewidth = 1.0, linestyle='--')
                    plt.scatter(line.T[0][0],line.T[1][0],s=50, marker='*', color ='skyblue')
                elif all:
                    plt.plot(line.T[0],line.T[1],color='red', linewidth = 1.0, linestyle='--')
                    plt.scatter(line.T[0][0],line.T[1][0],s=20, color ='red')
        
        # graph
        rot = data['rot'][0]
        orig = data['orig'][0]

        ctrs = data['graph'][0]['ctrs'][:, :2]
        ctrs = torch.matmul(ctrs, rot) + orig[:2]
        plt.scatter(ctrs.T[0], ctrs.T[1], c = 'black',s = 0.05)
       
        # trajectory history 
        indx = data['engage_indx'][0]
        target_indx = data['target_indx_e'][0]

        trajs = data['trajs_xyz'][0]
        masks = data['valid_masks'][0]

        trajs = [trajs[i] for i in indx]
        masks = [masks[i] for i in indx]

        if all == False:
            trajs = [trajs[i] for i in target_indx]
            masks = [masks[i] for i in target_indx]

        for i,traj in enumerate(trajs):
            traj = traj[:12][masks[i][:12]] 
            plt.plot(traj.T[0],traj.T[1],color='blue',linewidth=1.0, linestyle='--')
            
        
        plt.gca().set_aspect('equal')
        plt.title(msg)
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')
        plt.show()



def get_tarIndx(reg, data):
   
    indx = [x.clone().detach() for x in data['target_indx_e']]
    num = torch.tensor([len(x) for x in reg])
    num = torch.cumsum(num,0)[:-1]

    indx_cum = []
    indx_cum.append(indx[0])

    for ii in range(len(num)):
        indx_cum.append(indx[ii+1] + num[ii].item())

    indx_final = torch.cat(indx_cum,0)

    return indx_final



def panorama(data, path = False):

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

    if path:
        plt.savefig(path, dpi=2000)
    else:
        plt.show()

def plot_trajs(data):
    trajs = data['trajs_xyz']
    masks = data['valid_masks']
    for i in range(len(trajs)):
        line = trajs[i][masks[i]]
        plt.plot(line.T[0],line.T[1],color='blue',linewidth=1.0, linestyle='--')

    rot = data['rot']
    orig = data['orig']

    ctrs = data['graph']['ctrs'][:, :2]
    ctrs = np.matmul(ctrs, rot) + orig[:2]
    plt.scatter(ctrs.T[0], ctrs.T[1], c = 'black',s = 0.05)
    
    plt.gca().set_aspect('equal')
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')
    plt.show()