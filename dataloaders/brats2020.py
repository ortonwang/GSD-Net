import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
import random
from torch.utils.data.sampler import Sampler
import copy
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

def get_distance(img,max_ds=5):
    distance = distance_transform_edt(img)
    distance2 = distance_transform_edt(1 - img)
    distance[distance > max_ds] = max_ds
    distance2[distance2 > max_ds] = max_ds
    final_distance = distance + distance2
    return final_distance


class BaseDataSets_BraTS2020_no_distance(Dataset):
    def __init__(self, base_dir=None, split='train', data_txt="train.txt",  transform=None,
                 sup_type="label", ):
        self._base_dir = base_dir
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.sample_list = []

        data_path = self._base_dir + '/' + data_txt
        with open(data_path, 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        self.image_list = self.image_list#[:20]
        print("total {} samples for {}".format(len(self.image_list), self.split))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/{}".format(image_name), 'r')
        if self.split == "train":
            image = h5f['image'][:]
            gt = h5f['label'][:]
            label = h5f[self.sup_type][:]
            if self.sup_type == "noiseDE":
                label = label/255
            # if 255 in np.unique(label):
            #     new_label = copy.deepcopy(label)
            #     new_label[label == 255] = self.num_classes
            #     label = new_label
            sample = {'image': image, 'label': label.astype(np.uint8), "gt": gt.astype(np.uint8)}
            sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            gt = h5f['label'][:]
            sample = {'image': image, 'label': label.astype(np.uint8), "gt": gt.astype(np.uint8)}

        sample["idx"] = idx
        return sample

class BaseDataSets_BraTS2020_jit_flip_no_read(Dataset):
    def __init__(self, base_dir=None, split='train', data_txt="train.txt",  transform=None,
                 sup_type="label",max_dis=5 ):
        self._base_dir = base_dir
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.sample_list = []
        self.max_dis = max_dis
        s= 0.5
        self.jit =add_gaussian_noise

        data_path = self._base_dir + '/' + data_txt
        with open(data_path, 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        # self.image_list = self.image_list[:5]
        print("total {} samples for {}".format(len(self.image_list), self.split))
        self.tr_img_list,self.tr_gt_list,self.tr_label_list = [],[],[]
        for name in tqdm(self.image_list):
            h5f = h5py.File(self._base_dir + "/{}".format(name), 'r')
            self.tr_img_list.append(h5f['image'][:])
            self.tr_gt_list.append(h5f['label'][:])
            self.tr_label_list.append(h5f[self.sup_type][:])
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/{}".format(image_name), 'r')
        if self.split == "train":
            image = self.tr_img_list[idx] #h5f['image'][:]
            gt = self.tr_gt_list[idx]#h5f['label'][:]
            label = self.tr_label_list[idx]#h5f[self.sup_type][:]
            if self.sup_type == "noiseDE":
                label = label/255
            # if 255 in np.unique(label):
            #     new_label = copy.deepcopy(label)
            #     new_label[label == 255] = self.num_classes
            #     label = new_label
            sample = {'image': image, 'label': label.astype(np.uint8), "gt": gt.astype(np.uint8)}
            sample = self.transform(sample)
            distance_debug = get_distance(sample['label'],max_ds=self.max_dis)
            sample['distance'] = distance_debug
            sample['jit_image'] = self.jit(sample['image'])
            ratio = random.random()
            if ratio >0.5:
                sample['image'] = torch.flip(sample['image'],[1])
                sample['label'] = torch.flip(sample['label'],[0])

        else:
            image = self.tr_img_list[idx]#h5f['image'][:]
            label = self.tr_gt_list[idx]#h5f['label'][:]
            gt = self.tr_gt_list[idx]#h5f['label'][:]
            sample = {'image': image, 'label': label.astype(np.uint8), "gt": gt.astype(np.uint8)}

        sample["idx"] = idx
        return sample

class BaseDataSets_BraTS2020_jit_flip(Dataset):
    def __init__(self, base_dir=None, split='train', data_txt="train.txt",  transform=None,
                 sup_type="label",max_dis=5 ):
        self._base_dir = base_dir
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.sample_list = []
        self.max_dis = max_dis
        s= 0.5
        self.jit =add_gaussian_noise

        data_path = self._base_dir + '/' + data_txt
        with open(data_path, 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        # self.image_list = self.image_list[:5]
        print("total {} samples for {}".format(len(self.image_list), self.split))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/{}".format(image_name), 'r')
        if self.split == "train":
            image = h5f['image'][:]
            gt = h5f['label'][:]
            label = h5f[self.sup_type][:]
            if self.sup_type == "noiseDE":
                label = label/255
            # if 255 in np.unique(label):
            #     new_label = copy.deepcopy(label)
            #     new_label[label == 255] = self.num_classes
            #     label = new_label
            sample = {'image': image, 'label': label.astype(np.uint8), "gt": gt.astype(np.uint8)}
            sample = self.transform(sample)
            distance_debug = get_distance(sample['label'],max_ds=self.max_dis)
            sample['distance'] = distance_debug
            sample['jit_image'] = self.jit(sample['image'])
            ratio = random.random()
            if ratio >0.5:
                sample['image'] = torch.flip(sample['image'],[1])
                sample['label'] = torch.flip(sample['label'],[0])

        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            gt = h5f['label'][:]
            sample = {'image': image, 'label': label.astype(np.uint8), "gt": gt.astype(np.uint8)}

        sample["idx"] = idx
        return sample

class BaseDataSets_BraTS2020_jit_flip_SLIC(Dataset):
    def __init__(self, base_dir=None, split='train', data_txt="train.txt",  transform=None,
                 sup_type="label",max_dis=5 ):
        self._base_dir = base_dir
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.sample_list = []
        self.max_dis = max_dis
        s= 0.5
        self.jit =add_gaussian_noise
        self.path_SLIC = '/mnt/SSD250/orton/BraTS2020_resized_128_3/images_SLIC_4k/'

        data_path = self._base_dir + '/' + data_txt
        with open(data_path, 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        # self.image_list = self.image_list[:5]
        print("total {} samples for {}".format(len(self.image_list), self.split))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/{}".format(image_name), 'r')
        path_SLIC = self.path_SLIC + image_name.replace('Volumes','').replace('.h5','_flair.npy')
        if self.split == "train":
            image = h5f['image'][:]
            image_SLIC = np.load(path_SLIC)
            gt = h5f['label'][:]
            label = h5f[self.sup_type][:]
            if self.sup_type == "noiseDE":
                label = label/255
            # if 255 in np.unique(label):
            #     new_label = copy.deepcopy(label)
            #     new_label[label == 255] = self.num_classes
            #     label = new_label
            sample = {'image': image,'image_SLIC': image_SLIC, 'label': label.astype(np.uint8),"gt": gt.astype(np.uint8)}
            sample = self.transform(sample)
            if self.sup_type == 'noise04':
                distance_debug = np.load('/mnt/SSD250/orton/BraTS2020_resized_128_3/distance04/' +image_name.replace('Volumes','').replace('.h5','.npy' ))
            if self.sup_type == 'noise06':
                distance_debug = np.load('/mnt/SSD250/orton/BraTS2020_resized_128_3/distance06/' +image_name.replace('Volumes','').replace('.h5','.npy' ))
            if self.sup_type == 'noiseDE':
                distance_debug = np.load('/mnt/SSD250/orton/BraTS2020_resized_128_3/distanceDE/' +image_name.replace('Volumes','').replace('.h5','.npy' ))
            # distance_debug = get_distance(sample['label'],max_ds=self.max_dis)
            distance_debug = torch.from_numpy(distance_debug)
            sample['distance'] = distance_debug
            sample['jit_image'] = self.jit(sample['image'])
            ratio = random.random()
            if ratio >0.5:
                sample['image'] = torch.flip(sample['image'],[1])
                sample['image_SLIC'] = torch.flip(sample['image_SLIC'], [0])
                sample['label'] = torch.flip(sample['label'],[0])
                sample['distance'] = torch.flip(sample['distance'], [0])

        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            gt = h5f['label'][:]
            sample = {'image': image, 'label': label.astype(np.uint8), "gt": gt.astype(np.uint8)}

        sample["idx"] = idx
        return sample


class BaseDataSets_BraTS2020(Dataset):
    def __init__(self, base_dir=None, split='train', data_txt="train.txt",  transform=None,
                 sup_type="label", ):
        self._base_dir = base_dir
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.sample_list = []

        data_path = self._base_dir + '/' + data_txt
        with open(data_path, 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        self.image_list = self.image_list#[:20]
        print("total {} samples for {}".format(len(self.image_list), self.split))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/{}".format(image_name), 'r')
        if self.split == "train":
            image = h5f['image'][:]
            gt = h5f['label'][:]
            label = h5f[self.sup_type][:]
            if self.sup_type =='noiseDE':
                label = label/255
            # if 255 in np.unique(label):
            #     new_label = copy.deepcopy(label)
            #     new_label[label == 255] = self.num_classes
            #     label = new_label
            sample = {'image': image, 'label': label.astype(np.uint8), "gt": gt.astype(np.uint8)}
            sample = self.transform(sample)
            # distance_debug = get_distance(sample['label'].clone())
            sample['distance'] = get_distance(sample['label'].clone())
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            gt = h5f['label'][:]
            sample = {'image': image, 'label': label.astype(np.uint8), "gt": gt.astype(np.uint8)}

        sample["idx"] = idx
        return sample

def add_gaussian_noise(tensor, mean=0.05, std=0.02):
    """
    tensor: shape (C, H, W, D)，float32 类型
    mean: 噪声均值
    std: 噪声标准差
    """
    noise = torch.randn_like(tensor) * std + mean
    return tensor + noise

class BaseDataSets_BraTS2020_jit(Dataset):
    def __init__(self, base_dir=None, split='train', data_txt="train.txt",  transform=None,
                 sup_type="label", ):
        self._base_dir = base_dir
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.sample_list = []
        s= 0.5
        self.jit =add_gaussian_noise

        data_path = self._base_dir + '/' + data_txt
        with open(data_path, 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        self.image_list = self.image_list#[:20]
        print("total {} samples for {}".format(len(self.image_list), self.split))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/{}".format(image_name), 'r')
        if self.split == "train":
            image = h5f['image'][:]
            gt = h5f['label'][:]
            label = h5f[self.sup_type][:]
            if self.sup_type == "noiseDE":
                label = label/255
            # if 255 in np.unique(label):
            #     new_label = copy.deepcopy(label)
            #     new_label[label == 255] = self.num_classes
            #     label = new_label
            sample = {'image': image, 'label': label.astype(np.uint8), "gt": gt.astype(np.uint8)}
            sample = self.transform(sample)
            distance_debug = get_distance(sample['label'])
            sample['distance'] = distance_debug
            sample['jit_image'] = self.jit(sample['image'])
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            gt = h5f['label'][:]
            sample = {'image': image, 'label': label.astype(np.uint8), "gt": gt.astype(np.uint8)}

        sample["idx"] = idx
        return sample