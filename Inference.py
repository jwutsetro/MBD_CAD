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
from monai.transforms import AsDiscrete ,EnsureType,Compose
from monai.inferers import sliding_window_inference


from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.data.utils import pad_list_data_collate

from Utils import test_dicts, create_transforms_test, build_model
# importing functions from monai

from monai.inferers import sliding_window_inference
from monai.data import list_data_collate
from monai.data import DataLoader
from monai.networks.layers import Norm
from monai.networks.nets import DynUNet
from monai.losses import DiceLoss ,TverskyLoss
from monai.metrics import compute_meandice
from monai.utils import set_determinism

#python Inference.py --pretrain_file=/theia/scratch/brussel/vo/000/bvo00025/vsc10468/models/dataset_5_bis_split_0/BM_CV_final2021-12-16 17:23:00.538058_last --data_dir=/theia/scratch/brussel/vo/000/bvo00025/vsc10468/Dataset 
parser = argparse.ArgumentParser(description="Metastatic bone disease inference pipeline")
parser.add_argument("--pretrain_dir", default='nan', type=str, help="Path to file where the pretrained model is saved")
parser.add_argument("--data_dir", default="/dataset", type=str, help="dataset directory")
parser.add_argument("--output_dir_root", default="/output", type=str, help="Directory where output will be saved")
parser.add_argument("--preprocessing_type", default="5", type=int, help="type of preprocessing applied to the data in data_dir")
parser.add_argument("--test_time_augmentation", action="store_true", help="Wheter or not to use an ensemble for inference")
parser.add_argument("--sw_batch_size", default=6, type=int, help="number of sliding window batch size")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--save_proba", action="store_true", help="Wheter or not to save the probability output")
parser.add_argument("--save_mask", action="store_true", help="Wheter or not to save the binary mask")


def inferer(image,model,args):
    roi_size = (128,128,128)
    return sliding_window_inference(image, roi_size, args.sw_batch_size, model,overlap=0.33)

def infer(test_loader,args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(args)
    model.to(device)

    #loading the pretrained model 
    if args.pretrain_dir!='nan': 
        model.load_state_dict(torch.load(args.pretrain_dir,map_location=lambda storage, loc: storage))
    model.eval()
    
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, n_classes=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=2)])
    with torch.no_grad():
        for test_data in test_loader:
            test_inputs = test_data['image'].to(device) 
            image_name=test_data['image_meta_dict']['filename_or_obj'][0].split('/')[-2]
            print('Predicting MBD for patient: '+image_name)
            
            savermask = monai.data.NiftiSaver(output_dir=args.output_dir_root+'/mask',
                                              mode="nearest",separate_folder=False,output_postfix=image_name)
            saverpred=monai.data.NiftiSaver(output_dir=args.output_dir_root+'/pred',
                                            mode="bilinear",separate_folder=False,output_postfix=image_name)  
            
            n=0
            preds = torch.softmax(inferer(test_inputs,model,args), 1)
            n+=1
            if args.test_time_augmentation:
                for _ in range(3):
                    _img=RandGaussianNoised(['image'], prob=1.0, std=0.01)(test_data)['image']
                    pred = torch.softmax(inferer(_img.to(device), model,args), 1)
                    preds = preds + pred
                    n=n+1
                    for dims in [[0,1,2]]:
                        flip_pred = torch.softmax(inferer(torch.flip(_img.to(device), dims=dims), model,args), 1)
                        pred = torch.flip(flip_pred, dims=dims)
                        preds = preds + pred
                        n = n + 1.0
            preds = preds / n
        
           
            mask=(preds.argmax(dim=1, keepdims=True)).float()
            if args.save_mask:
                savermask.save_batch(mask, test_data["image_meta_dict"])
            if args.save_proba:
                saverpred.save_batch(preds, test_data["image_meta_dict"])

def main():
    args = parser.parse_args()
    args.in_channels=3+int(args.preprocessing_type>1)
    test_files=test_dicts(args)

    test_transforms=create_transforms_test(args)
    
    test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
    test_loader = monai.data.DataLoader(test_ds, batch_size=1, num_workers=args.workers)
    infer(test_loader,args)

if __name__ == '__main__':
    main()