import numpy as np
import torch
from data_loader import Waymo_Motion_DataLoader
from preprocess_6 import Waymo_Motion_Preprocess_6

config = dict()
config['train'] = '/home/avt/prediction/Waymo/dataset/train'
config['validation'] = '/home/avt/prediction/Waymo/dataset/validation'
config['pred_range'] = [-100.0, 100.0, -100.0, 100.0]
config['num_scales'] = 6
config['cross_dist'] = 6
config['downsample_factor'] = 10


def main():

    train_dataset = Waymo_Motion_DataLoader(config['train'])
    #train_dataset = Waymo_Motion_DataLoader(config['validation'])

    j = 49
    scen_list = train_dataset[j].read_TFRecord
    processed_list = Waymo_Motion_Preprocess_6(scen_list, config)
    for i,p in enumerate(processed_list):
        
        types = processed_list[i]['target_type']
        if 'TYPE_PEDESTRIAN' in types:
            print('Found a PEDESTRIAN in'+ str(j) + '_' + str(i) +'!')
            #torch.save(p,'/home/avt/prediction/Waymo/data_processed/train1/'+ str(i)+'.pt')
            torch.save(p,'/home/avt/prediction/Waymo/data_processed/train_p/'+ str(j) + '_' + str(i)  +'.pt')
    
if __name__ == "__main__":
    main()
    

