# From Noisy Labels to Intrinsic Structure: A Geometric-Structural Dual-Guided Framework for Noise-Robust Medical Image Segmentation
GSD-Net is a noise-robust medical image segmentation framework that mitigates the adverse effects of imperfect annotations.

[**[arXiv, 2025]**](https://arxiv.org/abs/2509.02419)


# Usage 
## Prepared data and Preprocess
To preprocess your datasets to generate simulated label noise, 
Please refer ./scripts/README.md and use ./scripts/generate_simulated_noise.py
### For generate superpixel Structural prior via SLIC,
Please refer ./scripts/README.md and use ./scripts/generate_SLIC.py

## Train model with noisy label using GSD-Net training
### Data prepare
Download the data from  [**[Google Drive]**](https://drive.google.com/file/d/10y0iXTcaN9lvYuR_DZB7JTVloqzpBvhJ/view?usp=sharing)
and unzip the data.zip

The ground truth for BUSUC dataset is derived from [**[Ahmed Iqbal et al.]**](https://www.kaggle.com/datasets/orvile/bus-uc-breast-ultrasound) 

Data Architecture:
```
├── data
│   ├── Kvasir
│   │   ├── images                                  # training images
│   │   ├── images_SLIC                             # generate super-pixel Structural prior 
│   │   ├── images_test                             # testing images
│   │   ├── masks                                   # ground truth for training images
│   │   ├── masks_200_0.2_0.05_0.2_mask_channel0    # simulated S_R label noise
│   │   ├── masks_200_0.8_0.05_0.2_mask_channel0    # simulated S_E label noise 
│   │   ├── masks_DE                                # simulated S_DE label noise
│   │   └── masks_test                              # ground truth for testing images

│   ├── BUSUC
│   ├── lidc
│   ├── MMIS2024TASK1
│   └── shenzhen
```
For our used Kvasir, Shenzhen, BU_SUC dataset:
```
sh train.sh
```
For training via multi-expert dataset (e.g. LIDC dataset and MMIS-2024 dataset)

For the LIDC-IDRI dataset, we use its pre-processed version as [**[MedicalMatting]**](https://link.springer.com/chapter/10.1007/978-3-030-87199-4_54)

For the MMIS-2024 dataset, we use the data from MMIS-2024 challenge in [**[ACM MM 2024]**](https://mmis2024.com/)
```
# For LIDC that each images with 4 annotations,random select 1 mask for training
python GSD_Net_lidc_each_select1.py

# For MMIS-2024 dataset 
python GSD_Net_MMIS2024_each_select1.py
```

## Training on other datasets
1. generate super-pixel Structural prior via ./scripts/generate_SLIC.py 
2. enter train_GSD-Net.py, function "data_path" in line 46, Set the folder path corresponding to the dataset
3. python train_GSD-Net.py

## Acknowledgements
Part of our code is adapted from [**[D-persona]**](https://github.com/ycwu1997/D-Persona)
Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.

## Questions
If any questions, feel free to contact me at 'ortonwangtao@gmail.com'

## ToDo 
README file needs further optimization
