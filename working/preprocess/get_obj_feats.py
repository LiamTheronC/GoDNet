import sys
sys.path.append('/home/avt/prediction/Waymo/working/')
import numpy as np
from utils import to_local


def get_obj_feats(data: dict) -> dict:

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

    data['engage_id'] = engage_id
    data['engage_indx'] = engage_indx
    data['feats'] = feats
    data['ctrs'] = ctrs
    data['gt_preds'] = gt_preds 
    data['has_preds'] = has_preds

    target_indx_e = np.array([list(engage_id).index(id) for id in data['target_id']])
    data['target_indx_e'] = target_indx_e
    
    return data
    





