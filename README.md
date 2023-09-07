# Computer-Aided Diagnosis of Skeletal Metastases in Multi-Parametric Whole-Body MRI
Official Implementation of "Computer-Aided Diagnosis of Skeletal Metastases in Multi-Parametric Whole-Body MRI""

**The full paper is under revision at :** [Computer Methods and Programs in Biomedicine ](https://www.sciencedirect.com/journal/computer-methods-and-programs-in-biomedicine)



## Architecture of network
The network architecture is made with the Dynamic Unet function of monai which is an implementation of the nnU-Net model wihtin monai. The exact network architecture is shown below.

<div align=center><img src="https://github.com/jwutsetro/MBD_CAD/blob/main/U-net_architecture.png" width="1000" height="500" alt="Result"/></div>

## Installation & Usage
### Instalation of monai 0.7
The scripts are build making use of monai 0.7. Recent versions of monai are not compatible with the code as syntax has changed. When installing monai to run the scripts, ensure to install the correct version. 
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

### Dataset
- To run the scripts, the dataset should be structered as indictated bellow. Note that each patient has 4 associated images to them: T1, b1000, ADC and mask. More clarification on the preprocessing steps applied to the image prior to running these scripts can be found in ( add reff later when paper is accapted) 
```
|-- Dataset
|   |-- Train
|   |   |-- Patient_1
|   |   |  |-- T1.nii
|   |   |  |-- b1000.nii
|   |   |  |-- ADC.nii
|   |   |  |-- mask.nii
|   |   |-- Patient_2
|   |   |  |-- T1.nii
|   |   |  |-- b1000.nii
|   |   |  |-- ADC.nii
|   |   |  |-- mask.nii
|   |-- Validation
|   |   |-- Patient_1
|   |   |  |-- T1.nii
|   |   |  |-- b1000.nii
|   |   |  |-- ADC.nii
|   |   |  |-- mask.nii
|   |-- Test
|   |   |-- Patient_1
|   |   |  |-- T1.nii
|   |   |  |-- b1000.nii
|   |   |  |-- ADC.nii
|   |   |  |-- mask.nii


```

The Train and Validation folder need to have patients in it to use the Train.py script. To use the Inference.py script, the Test folder should contain patients. 

### Preprocessing steps aaplied before training the different U-Nets. 
<div align=center><img src="https://github.com/jwutsetro/MBD_CAD/blob/main/Prepro_image.png" width="1000" alt="Result"/></div>
<div align=center><img src="https://github.com/jwutsetro/MBD_CAD/blob/main/Prepro_table.png" width="1000" alt="Result"/></div>

### Metastatic lesion Segmentation Figure Results
Quantitative Comparison in Learning Ability of the UNet model with incremental complexity of preprocessing. 
<div align=center><img src="https://github.com/jwutsetro/MBD_CAD/blob/main/lesions_zoom_corrected.png" width="1000" alt="Result"/></div>

### Saved U-Net Models
The saved U-Net models obtained in this work can be found  [here (Dynamic U-nets)](https://drive.google.com/drive/folders/1A3vIjswUFuXXESEiW-YdSL2jqyU2_SXL?usp=sharing).


### Citation
If you think this paper helps, please cite:
```
Fill in when paper is accepted
```
