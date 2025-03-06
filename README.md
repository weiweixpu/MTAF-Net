# MTAF-Net: Multi-Task Learning for Automated Glioma Segmentation and IDH Genotyping in Multi-parametric MRI

This is a PyTorch implementation of MTAF-Net proposed by our paper "MTAF-Net: Multi-Task Learning for Automated Glioma Segmentation and IDH Genotyping in Multi-parametric MRI.

## Requirements
Our models are trained and tested in the environment of PyTorch 1.9.1, NVIDIA NVIDIA 3090 GPU, CUDA 11.1. 

## Datasets

### Data available 
The data used in this study includes the MRI data and IDH genomic information. MRI data are derived from [BraTS 2020](https://ipp.cbica.upenn.edu/) and [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/), and the corresponding genomic information is from [The Cancer Genome Atlas](https://portal.gdc.cancer.gov/). We have provided the name mapping between BraTS 2020 and TCIA, which can been found in the data directory "MTAF-Net/data/".  

### Data preprocessing
We included multi-center datasets, including private datasets and publicly available datasets, so we applied image preprocessing techniques such as skull-stripped [HD-BET](https://github.com/MIC-DKFZ/HD-BET) algorithm, co-registering [FSL-flirt](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) algorithm, etc. to standardize all datasets.We have provided the MNI152 standard template, which can been found in the data directory "MTAF-Net/data/".

### MRI to PKL
Running the MRI_To_pkl.py file can package each patient's MR images, segmentation labels, and classification labels into a pkl file.
```bash
python3 MRI_To_PKL.py 
```

### Data preparation 
Datasets with the following folder structure.
```
│MRI/
├──train/
│  ├── ID1_Training_data.pkl
│  ├── ID2_Training_data.pkl
│  ├── ID3_Training_data.pkl
│  ├── ......
├──valid/
│  ├── ID1_Validation_data.pkl
│  ├── ID2_Validation_data.pkl
│  ├── ID3_Validation_data.pkl
│  ├── ......
├──test/
│  ├── BraTS20_Training_158_data.pkl
│  ├── BraTS20_Training_159_data.pkl
│  ├── ......
│  ├── BraTS20_Validation_033_data.pkl
│  ├── BraTS20_Validation_034_data.pkl
│  ├── ......
```
## Training the model
```bash
python3 train.py 
```
## model Weight
The model weights of [MTAF-Net](https://huggingface.co/WeiWei-XPU/best_network/tree/main) in our paper can be downloaded from Hugging Face and placed in the following path after downloading.
```bash
MTAF-Net/trails/best_network.pth.tar
```
## Testing the model
```bash
python3 test.py 
```
