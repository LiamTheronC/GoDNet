import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, distributed
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import random
from utils import collate_fn
import logging
from metrics.metrics import Postprocess
import time
from datetime import date
import argparse
from model.GoDNet import GreatNet 
from losses.godnet import Loss


def parse_args():
    parser = argparse.ArgumentParser(description='GoDNet training')
    parser.add_argument('--type-feats', 
                        choices=['vp', 'xyvp', 'xyz','xyp'], default='vp', 
                        help='types of feature, vp represents velocity and heading angle')
    parser.add_argument('--num-gpus', 
                        default=8, 
                        type=int, 
                        help='Number of GPUs to use for training')
    parser.add_argument('--resume-from', 
                        help='the checkpoint file to resume from')
    parser.add_argument('--lr', 
                        default=1e-3, 
                        type=float,
                        help='learning rate set for training')
    parser.add_argument('--batch-size', 
                        default=32, 
                        type=int,
                        help='batch size set for training')
    args = parser.parse_args()
  
    return args


class W_Dataset(Dataset):
    def __init__(self, path) -> None:

        self.path = path
        self.files = os.listdir(path)
    
    def __getitem__(self, index) -> dict:

        data_path = os.path.join(self.path,self.files[index])
        data = torch.load(data_path)

        return data
    
    def __len__(self) -> int:

        return len(self.files)


def train1(net,train_loader,loss_f,optimizer,epoch,num_epochs, post):
    # output of the net is directly trajectories.
    net.train()
    metrics = dict()
    start_time = time.time()
    for batch_idx, data in enumerate(train_loader):

        outputs = net(data)
        loss_out = loss_f(outputs,data)
        

        # Backward and optimize
        optimizer.zero_grad()
        loss_out.backward()
        optimizer.step()

        post.append(metrics,loss_out.item(),outputs,data)

    dt = time.time() - start_time
    post.display(metrics, dt, epoch, num_epochs, "Train")


def val1(net, val_loader, loss_f, epoch, num_epochs, post, Tfde_a, config):
    net.eval()
    metrics = dict()
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):

            outputs = net(data)
            loss_out = loss_f(outputs,data)
            post.append(metrics,loss_out.item(),outputs,data)
    
    dt = time.time() - start_time
    _, Tfde = post.display(metrics, dt, epoch, num_epochs, "Validation")

    if Tfde < Tfde_a:
        print('update weights')
        torch.save(net.state_dict(), 'weights/{config['name']}_{config['type_feats']}_{config['dd']}.pth')
        Tfde_a = Tfde

    return Tfde_a


def main():
    args = parse_args()
  
    if args.num_gpus > 1:
        if torch.distributed.is_available():
            torch.distributed.init_process_group(backend='nccl')
        else:
            print("Distributed training is not available. Exiting.")
            exit()
          
    script_dir = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.dirname(script_dir)
    
    seed = 33
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
  
    
  
    config = dict()
    config['n_actornet'] = 128
    config['num_epochs'] = 150
    config['lr'] = args.lr
    config["num_scales"] = 6
    config["n_map"] = 128
    config["n_actor"] = 128
    config['n_mark'] = 128
    config["actor2map_dist"] = 7.0 # 7.0
    config["map2actor_dist"] = 6.0 # 6.0
    config["actor2actor_dist"] = 50.0
    config["num_mods"] = 6
    config["pred_size"] = 80
    config["pred_step"] = 1
    config["num_preds"] = config["pred_size"] // config["pred_step"]
    config["cls_th"] = 2.0 #5.0, 2.0
    config["cls_th_2"] = 0.6
    config["cls_ignore"] = 0.2
    config["mgn"] = 0.2
    config["cls_coef"] = 1.0
    config["reg_coef"] = 1.0
    config["metrics_preds"] = 80
    config['batch_size'] = args.batch_size
    config['acrs'] = [20,40,60] # [40,80]
    config['cut'] = range(10,50)
    config["dim_feats"] = {'xyvp':[6,2], 'xyz':[4,3], 'xy':[3,2], 'xyp':[4,2], 'vp':[4,2], 'vpt':[5,2]}
    config['type_feats'] = args.type_feats
    config['name'] = 'GoDNet'
    config['train_split'] = os.path.join(working_dir, 'data_processed', 'train')
    config['val_split'] = os.path.join(working_dir, 'data_processed', 'val')
    config['dd'] = date.today().strftime('%m%d')
  
  
    net = GreatNet(config)
    if args.num_gpus > 1:
        if torch.cuda.device_count() >= args.num_gpus:
            device_ids = list(range(args.num_gpus))
            net = torch.nn.DataParallel(net, device_ids=device_ids)
        else:
            print("Error: Insufficient available GPUs!")
            exit()
    else:
        print("Error: need more GPUs")
        exit()
      
    if args.resume_from is not None:
        resume_path = os.path.join(working_dir, 'weights', args.resume_from)
        if os.path.isfile(resume_path):
            checkpoint = torch.load(resume_path) 
            net.load_state_dict(checkpoint)
        else:
            print(f"No checkpoint file found at {resume_path}. Exiting.")
            exit()
  
    net.cuda()
  
    loss_f = Loss(config)
    loss_f.cuda()
  
    post = Postprocess(config)
  
    optimizer = optim.Adam(net.parameters(), lr = config['lr'])
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.08)
  
    batch_size = config['batch_size']
    dataset_train = W_Dataset(config['train_split'])
    
    if args.num_gpus > 1 and torch.distributed.is_initialized():
        train_sampler = DistributedSampler(dataset_train)
    else:
        train_sampler = None
      
    train_loader = DataLoader(dataset_train, 
                           batch_size = batch_size,
                           sampler=train_sampler,
                           collate_fn = collate_fn, 
                           shuffle = True, 
                           drop_last=True)
    
    dataset_val = W_Dataset(config['val_split'])
    val_loader = DataLoader(dataset_val, 
                           batch_size = batch_size ,
                           collate_fn = collate_fn, 
                           shuffle = True, 
                           drop_last=True)
    
  
    num_epochs = config['num_epochs']
    
    log_file = 'train.log'
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=log_file, level=logging.INFO, format=log_format)
  
    msg = 'Training info: ' + config['name'] + ',' + config['type_feats']
    print(msg)
    logging.info(msg)
  
    Tfde_a = 100
    for epoch in range(num_epochs):
        train1(net,train_loader,loss_f,optimizer,epoch,num_epochs,post)
        #scheduler.step()
        if (epoch + 1) % 2 == 0 or epoch == 0:
            Tfde_a = val1(net,val_loader,loss_f,epoch,num_epochs, post, Tfde_a, config)


if __name__ == "__main__":
    main()
