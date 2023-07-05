import sys
sys.path.append('/home/avt/prediction/Waymo/working/')
import numpy as np
import torch
from dataLoader.data_loader import Waymo_Motion_DataLoader
from preprocess import Waymo_Motion_Preprocess

config = dict()
config['train'] = '/home/avt/prediction/Waymo/dataset/train'
config['val'] = '/home/avt/prediction/Waymo/dataset/validation'
config['pred_range'] = [-100.0, 100.0, -100.0, 100.0]
config['num_scales'] = 6
config['cross_dist'] = 6
config['downsample_factor'] = 10
config["dim_feats"] = {'xyvp':[6,2], 'xyz':[4,3], 'xy':[3,2], 'xyp':[4,2], 'vp':[4,2], 'vpt':[5,2]}
config['type_feats'] = 'vpt'
config['f'] = '1f'
config['aug'] = False


def main():

    path = config['train']

    if path == config['train']:
        word = 'train'
    elif path == config['val']:
        word = 'val'
    train_dataset = Waymo_Motion_DataLoader(path)
   
    j = 0
    scen_list = train_dataset[j].read_TFRecord
    processed_list = Waymo_Motion_Preprocess(scen_list, config)
    for i,p in enumerate(processed_list):
        # if i > 2:
        #     break
        torch.save(p,'/home/avt/prediction/Waymo/data_processed/'+ config['type_feats'] + '/'+ word +'_' + config['f'] + '/'+ str(j) + '_' + str(i)  +'.pt')
        

if __name__ == "__main__":
    main()
    

