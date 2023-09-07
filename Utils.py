import glob
import os


import monai
import torch
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityRanged, CropForegroundd, EnsureType,LoadImaged,RandCropByPosNegLabeld, RandAffined, ToTensord,RandFlipd, RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd, NormalizeIntensityd, AsDiscrete,MaskIntensityd,ConcatItemsd

from monai.data import DataLoader,CacheDataset,decollate_batch
from monai.networks.nets import DynUNet


  
def train_validate_dicts(args):
    data_dir_root=args.data_dir
    if args.preprocessing_type>1:
        train_files_T1=sorted(glob.glob(data_dir_root+'/Train/*/*T1*.nii'))
        train_files_b1000=sorted(glob.glob(data_dir_root+'/Train/*/*b1000*.nii'))
        train_files_ADC=sorted(glob.glob(data_dir_root+'/Train/*/*ADC*.nii'))
        train_files_mask=sorted(glob.glob(data_dir_root+'/Train/*/Whole_Body_GT_Man_Clean_r_resampled.nii'))
        train_files_skeleton=sorted(glob.glob(data_dir_root+'/Train/*/*Skeleton*.nii'))

        train_files=[{'image': [T1,B1000,ADC], 'mask': mask,'skeleton':skeleton}
          for T1,B1000,ADC,mask,skeleton in zip(train_files_T1,train_files_b1000,train_files_ADC,train_files_mask, train_files_skeleton)]


        val_files_T1=sorted(glob.glob(data_dir_root+'/Validation/*/*T1*.nii'))
        val_files_b1000=sorted(glob.glob(data_dir_root+'/Validation/*/*b1000*.nii'))
        val_files_ADC=sorted(glob.glob(data_dir_root+'/Validation/*/*ADC*.nii'))
        val_files_mask=sorted(glob.glob(data_dir_root+'/Validation/*/Whole_Body_GT_Man_Clean_r_resampled.nii'))
        val_files_skeleton=sorted(glob.glob(data_dir_root+'/Validation/*/*Skeleton*.nii'))

        val_files=[{'image': [T1,B1000,ADC], 'mask': mask,'skeleton':skeleton}
          for T1,B1000,ADC,mask,skeleton in zip(val_files_T1,val_files_b1000,val_files_ADC,val_files_mask, val_files_skeleton)]
        
    elif args.preprocessing_type==1:
        train_files_T1=sorted(glob.glob(data_dir_root+'/Train/*/T1.nii'))
        train_files_b1000=sorted(glob.glob(data_dir_root+'/Train/*/b1000.nii'))
        train_files_ADC=sorted(glob.glob(data_dir_root+'/Train/*/ADC.nii'))
        train_files_mask=sorted(glob.glob(data_dir_root+'/Train/*/Whole_Body_GT_Man_Clean_r_resampled.nii'))

        train_files=[{'image': [T1,B1000,ADC], 'mask': mask}
          for T1,B1000,ADC,mask in zip(train_files_T1,train_files_b1000,train_files_ADC,train_files_mask)]


        val_files_T1=sorted(glob.glob(data_dir_root+'/Validation/*/T1.nii'))
        val_files_b1000=sorted(glob.glob(data_dir_root+'/Validation/*/b1000.nii'))
        val_files_ADC=sorted(glob.glob(data_dir_root+'/Validation/*/ADC.nii'))
        val_files_mask=sorted(glob.glob(data_dir_root+'/Validation/*/Whole_Body_GT_Man_Clean_r_resampled.nii'))


        val_files=[{'image': [T1,B1000,ADC], 'mask': mask}
          for T1,B1000,ADC,mask in zip(val_files_T1,val_files_b1000,val_files_ADC,val_files_mask)]
    print(train_files)
    print()
    print()
    print()
    print(val_files)
    return train_files ,val_files

def test_dicts(args):
    data_dir_root=args.data_dir
    if args.preprocessing_type>1:
        test_files_T1=sorted(glob.glob(data_dir_root+'/Validation/*/*T1*.nii'))
        test_files_b1000=sorted(glob.glob(data_dir_root+'/Validation/*/*b1000*.nii'))
        test_files_ADC=sorted(glob.glob(data_dir_root+'/Validation/*/*ADC*.nii'))
        test_files_skeleton=sorted(glob.glob(data_dir_root+'/Validation/*/*Skeleton*.nii'))

        test_files=[{'image': [T1,B1000,ADC],'skeleton':skeleton}
          for T1,B1000,ADC,skeleton in zip(test_files_T1,test_files_b1000,test_files_ADC, test_files_skeleton)]
        
    elif args.preprocessing_type==1:
        test_files_T1=sorted(glob.glob(data_dir_root+'/Validation/*/T1.nii'))
        test_files_b1000=sorted(glob.glob(data_dir_root+'/Validation/*/b1000.nii'))
        test_files_ADC=sorted(glob.glob(data_dir_root+'/Validation/*/ADC.nii'))
 

        test_files=[{'image': [T1,B1000,ADC]}
          for T1,B1000,ADC in zip(test_files_T1,test_files_b1000,test_files_ADC)]
        

    return test_files

def create_transforms(args):
    if args.preprocessing_type>1:
        train_transforms = [
            LoadImaged(keys=['image', 'mask','skeleton']),
            AddChanneld(keys=['mask','skeleton']),
            MaskIntensityd(keys=['image', 'mask'],mask_key='skeleton'),
            NormalizeIntensityd(keys=['image'],channel_wise=True,nonzero=True),
            ConcatItemsd(keys=["image", "skeleton"], name="image"),
            RandCropByPosNegLabeld(keys=['image', 'mask','skeleton'], label_key='mask',image_key='skeleton',image_threshold=0.1,spatial_size=(130,130,130), num_samples=args.sw_batch_size),
            RandAffined(
                        keys=['image', 'mask'],
                        prob=0.5,
                        rotate_range=(-0.05, 0.05),
                        scale_range=(-0.1, 0.1),
                        mode=("bilinear", "nearest"),
                        spatial_size=(128, 128,128),
                        as_tensor_output=False,
                    ),
                    RandGaussianSmoothd(keys=["image"],
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15),
                prob=0.3,
            ),
                    RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.01),
                    RandFlipd(keys=['image', 'mask'], spatial_axis=0, prob=0.5),
            RandFlipd(keys=['image', 'mask'], spatial_axis=1, prob=0.5),
            RandFlipd(keys=['image', 'mask'], spatial_axis=2, prob=0.5),
            ToTensord(keys=['image', 'mask'])]

        val_transforms=[
            LoadImaged(keys=['image', 'mask','skeleton']),
            AddChanneld(keys=['mask','skeleton']),
            MaskIntensityd(keys=['image', 'mask'],mask_key='skeleton'),

            NormalizeIntensityd(keys=['image'],channel_wise=True,nonzero=True),
            ConcatItemsd(keys=["image", "skeleton"], name="image"),
            ToTensord(keys=['image', 'mask'])]
        
    elif args.preprocessing_type==1:
        train_transforms = [
            LoadImaged(keys=['image', 'mask']),
            AddChanneld(keys=['mask']),
            NormalizeIntensityd(keys=['image'],channel_wise=True,nonzero=True),
            RandCropByPosNegLabeld(keys=['image', 'mask'], label_key='mask',spatial_size=(130,130,130), num_samples=args.sw_batch_size),
            RandAffined(
                        keys=['image', 'mask'],
                        prob=0.5,
                        rotate_range=(-0.05, 0.05),
                        scale_range=(-0.1, 0.1),
                        mode=("bilinear", "nearest"),
                        spatial_size=(128, 128,128),
                        as_tensor_output=False,
                    ),
                    RandGaussianSmoothd(keys=["image"],
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
                sigma_z=(0.5, 1.15),
                prob=0.3,
            ),
                    RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
                    RandGaussianNoised(keys=["image"], prob=0.3, std=0.01),
                    RandFlipd(keys=['image', 'mask'], spatial_axis=0, prob=0.5),
            RandFlipd(keys=['image', 'mask'], spatial_axis=1, prob=0.5),
            RandFlipd(keys=['image', 'mask'], spatial_axis=2, prob=0.5),
            ToTensord(keys=['image', 'mask'])]

        val_transforms=[
            LoadImaged(keys=['image', 'mask']),
            AddChanneld(keys=['mask']),
            NormalizeIntensityd(keys=['image'],channel_wise=True,nonzero=True),
            ToTensord(keys=['image', 'mask'])]
      
    return Compose(train_transforms) ,Compose(val_transforms)

def create_transforms_test(args):
    if args.preprocessing_type>1:

#         val_transforms=[
#             LoadImaged(keys=['image','skeleton']),
#             AddChanneld(keys=['skeleton']),
#             MaskIntensityd(keys=['image'],mask_key='skeleton'),
#             NormalizeIntensityd(keys=['image'],channel_wise=True,nonzero=True),
#             ConcatItemsd(keys=["image", "skeleton"], name="image"),
#             ToTensord(keys=['image'])]
        
        val_transforms=[
            LoadImaged(keys=['image','skeleton']),
            AddChanneld(keys=['skeleton']),
            MaskIntensityd(keys=['image'],mask_key='skeleton'),

            NormalizeIntensityd(keys=['image'],channel_wise=True,nonzero=True),
            ConcatItemsd(keys=["image", "skeleton"], name="image"),
            ToTensord(keys=['image'])]
        
    elif args.preprocessing_type==1:
        val_transforms=[
            LoadImaged(keys=['image']),
            NormalizeIntensityd(keys=['image'],channel_wise=True,nonzero=True),
            ToTensord(keys=['image'])]
            
    return Compose(val_transforms)


def build_model(args):
    strides, kernels = [], []
    spacings=[1,1,1]
    sizes=(128, 128,128)
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])

    model = DynUNet(
    spatial_dims=3,
    in_channels=args.in_channels,
    out_channels=2,
    kernel_size=kernels,
    strides=strides,
    upsample_kernel_size=strides[1:],
        dropout=0.3,
    deep_supervision=True,
    deep_supr_num=3,
    res_block=True,
        )

    return model 

post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=True, n_classes=2)])
post_mask = Compose([EnsureType(), AsDiscrete(to_onehot=True, n_classes=2)])



def _compute_loss(loss_function,preds, mask):
    pred=torch.unbind(preds,dim=1)
    return sum([0.5 ** i * loss_function(p,mask) for i, p in enumerate(pred)])/1.875


            