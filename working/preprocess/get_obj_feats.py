# The feats can be xy, xyz ,xyvp

import sys
sys.path.append('/home/avt/prediction/Waymo/working/')
import numpy as np
from utils import to_local


def get_obj_feats(data: dict, type_feats = 'xyvp') -> dict:

    # type_feats indicates different type of feats, 'xyvp' or 'xyz' or 'xy'

    orig = data['trajs_xyz'][data['sdc_index']][data['current_time_index']]
    pre_orig = data['trajs_xyz'][data['sdc_index']][data['current_time_index']-1]
    
    dir_vec = pre_orig - orig
    
    theta = np.pi - np.arctan2(dir_vec[1], dir_vec[0])
    rot = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]], np.float32)

    data['orig'] = orig
    data['theta'] = theta
    data['rot'] = rot

    feats, ctrs, gt_preds, has_preds, engage_id, engage_indx = [], [], [], [], [], []
    
    if type_feats == 'xyz':

        for i in range(len(data['object_ids'])):

            feat = np.zeros((11, 4), np.float32)
            traj_xyz = data['trajs_xyz'][i][:11]

            mask_i = data['valid_masks'][i][:11] 

            if mask_i[-1] != True:
                continue

            reverse = list(np.flip(mask_i))
            if False in reverse:
                index = -reverse.index(False)
                
                traj_xyz = traj_xyz[index:,]

                feat[index:,:3] = to_local(traj_xyz, orig, theta)
                feat[index:,3] = 1.0

            else:
                index = 0
                feat[:,:3] = to_local(traj_xyz, orig, theta)
                feat[:,3] = 1.0

            mask_gt = np.arange(11,91)
            gt_pred = data['trajs_xyz'][i][mask_gt]

            has_pred = data['valid_masks'][i][mask_gt]

            ctrs.append(feat[-1, :3].copy())
            feat[1:, :3] -= feat[:-1, :3]
            feat[index, :3] = 0

            feats.append(feat) 
            engage_id.append(data['object_ids'][i])
            engage_indx.append(i)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)
    

    elif type_feats == 'xy':
        
        for i in range(len(data['object_ids'])):

            feat = np.zeros((11, 3), np.float32)
            traj_xy = data['trajs_xyz'][i][:11,:2]
            mask_i = data['valid_masks'][i][:11] 

            if mask_i[-1] != True:
                continue

            reverse = list(np.flip(mask_i))
            if False in reverse:
                index = -reverse.index(False)
                
                traj_xy = traj_xy[index:,]

                feat[index:,:2] = to_local(traj_xy, orig, theta)
                feat[index:,2] = 1.0

            else:
                index = 0
                feat[:,:2] = to_local(traj_xy, orig, theta)
                feat[:,2] = 1.0

            mask_gt = np.arange(11,91)
            gt_pred = data['trajs_xyz'][i][mask_gt]

            has_pred = data['valid_masks'][i][mask_gt]

            ctrs.append(feat[-1, :2].copy())
            feat[1:, :2] -= feat[:-1, :2]
            feat[index, :2] = 0

            feats.append(feat) 
            engage_id.append(data['object_ids'][i])
            engage_indx.append(i)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        

    elif type_feats == 'xyvp':

        for i in range(len(data['object_ids'])):

            feat = np.zeros((11, 6), np.float32)
            traj_xy = data['trajs_xyz'][i][:11,:2]
            vel = data['velocity_xy_heading'][i][:11,:2]
            heading = data['velocity_xy_heading'][i][:11,2]
            mask_i = data['valid_masks'][i][:11] 

            if mask_i[-1] != True:
                continue

            reverse = list(np.flip(mask_i))
            if False in reverse:
                index = -reverse.index(False)
                
                traj_xy = traj_xy[index:,]
                vel = vel[index:,]
                heading = heading[index:,]


                feat[index:,:2] = to_local(traj_xy, orig, theta)
                feat[index:,2:4] = to_local(vel,np.array([0.0,0.0]),theta)
                feat[index:,4]= heading + theta
                feat[index:,5] = 1.0

            else:
                index = 0
                feat[:,:2] = to_local(traj_xy, orig, theta)
                feat[:,2:4] = to_local(vel,np.array([0.0,0.0]),theta)
                feat[:,4]= heading + theta
                feat[:,5] = 1.0

            mask_gt = np.arange(11,91)
            gt_pred = data['trajs_xyz'][i][mask_gt][:2]

            has_pred = data['valid_masks'][i][mask_gt]

            ctrs.append(feat[-1, :2].copy())
            feat[1:, :2] -= feat[:-1, :2]
            feat[index, :2] = 0

            feats.append(feat) 
            engage_id.append(data['object_ids'][i])
            engage_indx.append(i)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)
    

    data['engage_id'] = engage_id
    data['engage_indx'] = engage_indx
    data['feats'] = feats
    data['ctrs'] = ctrs
    data['gt_preds'] = gt_preds 
    data['has_preds'] = has_preds

    target_indx_e = np.array([list(engage_id).index(id) for id in data['target_id']])
    data['target_indx_e'] = target_indx_e
    
    return data
    





