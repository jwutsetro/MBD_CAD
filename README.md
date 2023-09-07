# This GitHub page is still under construction. The trained models will only be made available after the publication of the work. 

# Computer-Aided Diagnosis of Skeletal Metastases in Multi-Parametric Whole-Body MRI
Official Implementation of "Computer-Aided Diagnosis of Skeletal Metastases in Multi-Parametric Whole-Body MRI"

**The full paper is under revision at :** [Computer Methods and Programs in Biomedicine ](https://www.sciencedirect.com/journal/computer-methods-and-programs-in-biomedicine)



## Architecture of network
The network architecture is made with the Dynamic Unet function of monai, which implements the nnU-Net model within monai. The exact network architecture is shown below.

<div align=center><img src="https://github.com/jwutsetro/MBD_CAD/blob/main/U-net_architecture.png" width="1000" height="500" alt="Result"/></div>

## Installation & Usage
### Instalation of monai 0.7
The scripts are built making use of monai 0.7. Recent Monai versions are incompatible with the code as syntax has changed. When installing Monai to run the scripts, ensure to install the correct version. 
- Install monai (version 0.7.0) :
```
pip install monai==0.7
```

### Hardware requirements
The models are trained on a 40Gb A100 GPU. To ensure smooth dataloading, at least 12 CPU cores should be available on the system. 
### Running the scripts
- Running the training script with default parameters:
```
python Train.py --logdir=/path/to/logdir  --data_dir=/path/to/dataset
```
- Running the Inference script with default parameters and saving the binary predictions:
```
python Inference.py --pretrain_dir=/path/to/model_best.pt --data_dir=/path/to/Dataset --output_dir_root=/path/to/results --save_mask
```

## Dataset
- The dataset should be structured as indicated below to run the scripts. Note that each patient has five associated images: T1, b1000, ADC and mask, and a skeleton mask. More clarification on the preprocessing steps applied to the dataset before running the scripts can be found in (add reference later when the paper is published) 
```
|-- Dataset
|   |-- Train
|   |   |-- Patient_1
|   |   |  |-- T1.nii
|   |   |  |-- b1000.nii
|   |   |  |-- ADC.nii
|   |   |  |— mask.nii
|   |   |  |— skeleton.nii
|   |   |-- Patient_2
|   |   |  |-- T1.nii
|   |   |  |-- b1000.nii
|   |   |  |-- ADC.nii
|   |   |  |— mask.nii
|   |   |  |— skeleton.nii
|   |-- Validation
|   |   |-- Patient_1
|   |   |  |-- T1.nii
|   |   |  |-- b1000.nii
|   |   |  |-- ADC.nii
|   |   |  |— mask.nii
|   |   |  |— skeleton.nii
|   |-- Test
|   |   |-- Patient_1
|   |   |  |-- T1.nii
|   |   |  |-- b1000.nii
|   |   |  |-- ADC.nii
|   |   |  |— mask.nii
|   |   |  |— skeleton.nii


```

The Train and Validation folder must have patients in it to use the Train.py script. To use the Inference.py script, the Test folder should contain patients. 

## Preprocessing steps applied before training the different U-Nets. 
Different preprocessing steps can be applied before using the training and inference scripts provided on this page. When using the pre-trained models that are available [here (Dynamic U-nets)](https://drive.google.com/drive/folders/1A3vIjswUFuXXESEiW-YdSL2jqyU2_SXL?usp=sharing), it is important to select a model that is trained with the same preprocessing applied to it as your data.  The table below describes the different preprocessing steps applied to models 1-5. For more information on the preprocessing, please have a look at the journal publication. 
<div align=center><img src="https://github.com/jwutsetro/MBD_CAD/blob/main/Prepro_image.png" width="1000" alt="Result"/></div>
<div align=center><img src="https://github.com/jwutsetro/MBD_CAD/blob/main/Prepro_table.png" width="1000" alt="Result"/></div>

## Metastatic lesion Segmentation Figure Results
Quantitative Comparison of the U-Net models with incremental preprocessing complexity. Best results are obtained when using the most complex preprocessing scheme (5) 
<div align=center><img src="https://github.com/jwutsetro/MBD_CAD/blob/main/lesions_zoom_corrected.png" width="1000" alt="Result"/></div>

## Saved U-Net Models
The saved U-Net models obtained in this work can be found  [here (Dynamic U-nets)](https://drive.google.com/drive/folders/1A3vIjswUFuXXESEiW-YdSL2jqyU2_SXL?usp=sharing).


## Citation
If you think this paper helps, please cite:
```
Fill in when the paper is published
```
