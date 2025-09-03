import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler

#   A.IAAAdditiveGaussianNoise(scale=(0.01*255, 0.05*255), p=0.5)
#   A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5)
#   A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
#   Poisson噪声
#   cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2, 0.1), rotate_limit=40, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.5,
            contrast_limit=0.1,
            p=0.5
        ),
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=100, val_shift_limit=80),
        A.OneOf([
            # A.CoarseDropout(max_holes=100, max_height=aug_size, max_width=aug_size, fill_value=[239, 234, 238]),
            A.GaussNoise()
        ]),
        A.OneOf([
            A.ElasticTransform(),
            A.GridDistortion(),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0)
        ]),
        ToTensorV2()
        ], p=1.)

train_transform2 = A.Compose([
        A.RandomRotate90(),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2, 0.1), rotate_limit=40, p=0.5),
        # A.RandomBrightnessContrast(
        #     brightness_limit=0.5,
        #     contrast_limit=0.1,
        #     p=0.5
        # ),
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=100, val_shift_limit=80),
        # A.OneOf([
            # A.CoarseDropout(max_holes=100, max_height=aug_size, max_width=aug_size, fill_value=[239, 234, 238]),
            # A.GaussNoise()
        # ]),
        # A.OneOf([
        #     A.ElasticTransform(),
        #     A.GridDistortion(),
        #     A.OpticalDistortion(distort_limit=0.5, shift_limit=0)
        # ]),
        ToTensorV2()
        ], p=1.)
test_transform = A.Compose([
     ToTensorV2()], p=1.)


class train_dataset(Dataset):
    def __init__(self, root_path, names_list_img,names_list_gt, transform):
        self.root_path = root_path
        self.names_list_img = names_list_img
        self.names_list_gt = names_list_gt
        self.transforms = transform

    def __getitem__(self, index):
        # A_path = self.root_path + self.names_list_img[index]
        # B_path = self.root_path + self.names_list_gt[index]
        img = np.load(self.root_path + self.names_list_img[index])
        gt = np.load(self.root_path + self.names_list_gt[index])
        # weights = 0
        img = self.transforms(image=img, mask=gt)

        return img['image'], img['mask']
        # return {'A': img['image'], 'B': img['mask'], 'A_paths': A_path, 'B_paths': B_path, 'weights': weights}
    def __len__(self):
        return len(self.names_list_gt)

class train_dataset2(Dataset):
    def __init__(self, root_path, names_list_img,names_list_gt, transform):
        self.root_path = root_path
        self.names_list_img = names_list_img
        self.names_list_gt = names_list_gt
        self.transforms = transform

    def __getitem__(self, index):
        img = np.load('/media/orton_SSD/YGA_CT/Train/' + self.names_list_img[index],allow_pickle=True)
        gt = np.load('/media/orton_iacc2024/Train/'+ self.names_list_gt[index],allow_pickle=True)
        # gt = np.load(self.root_path + self.names_list_gt[index])

        img = self.transforms(image=img, mask=gt)

        return img['image'], img['mask']

    def __len__(self):
        return len(self.names_list_gt)



class train_dataset3(Dataset):
    def __init__(self, root_path, names_list_img,names_list_gt, transform):
        self.root_path = root_path
        self.names_list_img = names_list_img
        self.names_list_gt = names_list_gt
        self.transforms = transform

    def __getitem__(self, index):
        image_np = np.load('/media/orton_SSD/YGA_CT/Train/' + self.names_list_img[index],allow_pickle=True)
        gt = np.load('/media/orton_iacc2024/Train/'+ self.names_list_gt[index],allow_pickle=True)
        # gt = np.load(self.root_path + self.names_list_gt[index])
        poisson_scale = 2048

        if random.random() < 0.5:
            gaussian_noise = np.random.normal(0, 0.1, image_np.shape)
            image_np += gaussian_noise
            image_np = np.clip(image_np, 0, 1)

        # 2. 添加泊松噪声
        if random.random() < 0.5:
            scaled_image = image_np * poisson_scale
            poisson_noise = np.random.poisson(scaled_image)
            image_np = poisson_noise / poisson_scale
            image_np = np.clip(image_np, 0, 1)
        img = self.transforms(image=image_np, mask=gt)

        return img['image'], img['mask']

    def __len__(self):
        return len(self.names_list_gt)

class train_dataset4(Dataset):
    def __init__(self, root_path, names_list_img,names_list_gt, transform):
        self.root_path = root_path
        self.names_list_img = names_list_img
        self.names_list_gt = names_list_gt
        self.transforms = transform

    def __getitem__(self, index):
        image_np = np.load('/mnt/public/1_public_dataset/YGA_CT/Train/' +
                           self.names_list_img[index].replace('train_lowdose_npy_orton','train_lowdose_npy_orton_ori'),allow_pickle=True)
        # gt = np.load('/mnt/public/1_public_dataset/YGA_CT/Train/'+
        #              self.names_list_gt[index].replace('train_label_npy_orton','train_label_npy_orton_ori'),allow_pickle=True)
        gt = np.load('/media/orton_iacc2024/Train/' + self.names_list_gt[index], allow_pickle=True)
        # gt = np.load(self.root_path + self.names_list_gt[index])

        if random.random() < 0.5:
            # scaled_image = image_np * poisson_scale
            min_s = np.min(image_np)
            image_np = np.random.poisson(image_np-min_s)+min_s
            # image_np = poisson_noise / poisson_scale
            # image_np = np.clip(image_np, 0, 1)
        if random.random() < 0.5:
            min_s = np.min(image_np)
            image_np = np.random.poisson(image_np-min_s)+min_s

        if random.random() < 0.5:
            min_s = np.min(image_np)
            image_np = np.random.poisson(image_np-min_s)+min_s

        min, max = -1024, 1024
        image_np = np.clip(image_np, min, max)
        image_np = (image_np - min) / (max - min)  # normalize
        # # 2. 添加泊松噪声
        # if random.random() < 0.5:
        #     scaled_image = image_np * poisson_scale
        #     poisson_noise = np.random.poisson(scaled_image)
        #     image_np = poisson_noise / poisson_scale
        #     image_np = np.clip(image_np, 0, 1)
        if random.random() < 0.5:
            gaussian_noise = np.random.normal(0, 0.1, image_np.shape)
            image_np += gaussian_noise
            image_np = np.clip(image_np, 0, 1)

        img = self.transforms(image=image_np, mask=gt)

        return img['image'], img['mask']

    def __len__(self):
        return len(self.names_list_gt)


class infer_dataset(Dataset):
    def __init__(self, root_path, names_list_img, transform):
        self.root_path = root_path
        self.names_list_img = names_list_img
        self.transforms = transform

    def __getitem__(self, index):
        path = self.root_path + self.names_list_img[index]
        img = np.load(path)

        img = self.transforms(image=img)

        return img['image'],path

    def __len__(self):
        return len(self.names_list_img)