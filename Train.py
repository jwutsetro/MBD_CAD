import os
import glob
import numpy as np
import time
from datetime import datetime
import argparse
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import monai
from monai.transforms import AsDiscrete,EnsureType,Compose
from monai.inferers import sliding_window_inference


from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.data.utils import pad_list_data_collate
from monai.data import DataLoader,decollate_batch
from Utils import train_validate_dicts, create_transforms, build_model, _compute_loss


parser = argparse.ArgumentParser(description="Metastatic bone disease segmentation pipeline")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the trained model")
parser.add_argument("--data_dir", default="/dataset", type=str, help="dataset directory")
parser.add_argument("--preprocessing_type", default="5", type=int, help="type of preprocessing applied to the data in data_dir")

parser.add_argument("--val_every", default=20,type=int, help="validation frequency")
parser.add_argument("--max_epochs", default=1000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-3, type=float, help="optimization learning rate")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--workers", default=8, type=int, help="number of workers")

parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--smooth_dr", default=1e-5, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=1e-5, type=float, help="constant added to dice numerator to avoid zero")







def train_process(train_ds,val_ds,args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_metric=0
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              collate_fn=pad_list_data_collate)
    val_loader=DataLoader(val_ds,batch_size=1,shuffle=False,
                          num_workers=args.workers,
                          collate_fn=pad_list_data_collate)
    
    model = build_model(args)
    model.to(device)
    
    
    loss_function = DiceCELoss(include_background=False,to_onehot_y=True, softmax=True,smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr,lambda_dice=0.5,lambda_ce=0.5)
        

    optimizer=torch.optim.SGD(model.parameters(), lr=args.optim_lr, momentum=args.momentum,nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda epoch: (1 - epoch / args.max_epochs) ** 0.9)
    dice_metric=DiceMetric(include_background=False,reduction='mean')
    

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, n_classes=2)])
    post_mask = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=2)])

    for epoch in range(args.max_epochs):

        epoch_start = time.time()
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, args.max_epochs))
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:

            step_start = time.time()
            step += 1
            inputs, masks = batch_data['image'].to(device), batch_data['mask'].to(device)
            masks=(masks==1).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = _compute_loss(loss_function,outputs, masks)


            print('%d/%d, train_loss: %0.4f, step time: %0.4f' %
                  (step, len(train_ds) // train_loader.batch_size, loss.item(), time.time() - step_start))

            epoch_len=len(train_ds)//train_loader.batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len=len(train_ds)//train_loader.batch_size



        
        scheduler.step()
        epoch_loss /= step

        lr=scheduler.get_last_lr()
        del inputs
        del masks
        del outputs

        # validation loop for every 5 eppochs 
        if (epoch + 1) % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_masks = val_data['image'].to(device),val_data['mask'].to(device)
                    val_masks=(val_masks==1).float()
                    
                    val_outputs = sliding_window_inference(val_inputs, (256, 256, 256), 1, model,overlap=0.5,mode='gaussian')
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_masks = [post_mask(i) for i in decollate_batch(val_masks)]
                    
                    metric=dice_metric(
                        y_pred=val_outputs,
                        y=val_masks).item()
                    
                    
                    metric_sum+=metric
                    metric_count+=1
                    dice_metric.reset()
                metric = metric_sum/metric_count
                if metric>best_metric:
                    best_metric=metric
                    torch.save(model.state_dict(), args.logdir+'/model_best.pt')
                torch.save(model.state_dict(), args.logdir+'/model_last.pt')
              

                del val_masks
                del val_outputs
                del metric
                del val_inputs

def main():
    # setting up the environment 
    args = parser.parse_args()
    args.in_channels=3+int(args.preprocessing_type>1)
    
    
    
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)
    
    # creating the dataloaders
    train_files,val_files = train_validate_dicts(args)
    train_transforms,val_transforms = create_transforms(args)
    train_ds = monai.data.CacheDataset(data = train_files, transform = train_transforms,cache_rate=1.0, num_workers=args.workers)
    val_ds=monai.data.CacheDataset(data = val_files, transform = val_transforms,cache_rate=1.0, num_workers=args.workers)

    # start train process
    train_process(train_ds,val_ds,args)
        
if __name__ == '__main__':
    main()