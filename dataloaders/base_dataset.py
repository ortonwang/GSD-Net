import cv2
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt
train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(p=0.5),
        ToTensorV2()
        ], p=1.)
train_transform_npc = A.Compose([
    A.Resize(128, 128,interpolation=cv2.INTER_NEAREST),
    A.RandomRotate90(),
    A.Flip(p=0.5),
    ToTensorV2()
], p=1.)
test_transform = A.Compose([
     ToTensorV2()], p=1.)
test_transform_npc = A.Compose([
    A.Resize(128, 128,interpolation=cv2.INTER_NEAREST),
    ToTensorV2()], p=1.)

class train_dataset_noread_distance_jit_SLIC(Dataset):
    def __init__(self, imgs,gts,SLIC,transform,datasets_h='kvasir'):
        self.imgs = imgs
        self.gts = gts
        self.SLIC = SLIC
        self.transforms = transform
        self.datasets_h = datasets_h

        if self.datasets_h != 'BUSUC':s = 0.5
        else:s = 0.05
        self.jit = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    def __getitem__(self, index):

        img = self.imgs[index]
        gt = self.gts[index]
        img_SLIC = self.SLIC[index]
        gt = gt.astype(int)
        #weights = 0
        img = self.transforms(image=img.copy(),image2=img_SLIC.copy(), mask=gt.copy())
        distance_debug = get_distance(img['mask'])
        return img['image'],self.jit(img['image']), img['mask'], torch.from_numpy(distance_debug), img['image2']
    def __len__(self):
        return len(self.gts)

class train_dataset_noread_distance_jit_SLIC_lidc(Dataset):
    def __init__(self, imgs,gts,SLIC,transform):
        self.imgs = imgs
        self.gts = gts
        self.SLIC = SLIC
        self.transforms = transform
        s = 0.05
        self.jit = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

    def __getitem__(self, index):
        # A_path = self.root_path + self.names_list_img[index]
        # B_path = self.root_path + self.names_list_gt[index]
        img = self.imgs[index]
        gt = self.gts[index]
        img_SLIC = self.SLIC[index]
        gt = gt.astype(int)
        #weights = 0

        img = self.transforms(image=img.copy(),image2=img_SLIC.copy(), mask=gt.copy())
        distance_debug = get_distance(img['mask'])
        return img['image'],self.jit(img['image']), img['mask'], torch.from_numpy(distance_debug), img['image2']
        # return {'A': img['image'], 'B': img['mask'], 'A_paths': A_path, 'B_paths': B_path, 'weights': weights}
    def __len__(self):
        return len(self.gts)

class train_dataset_noread(Dataset):
    def __init__(self, imgs,gts,transform):
        self.imgs = imgs
        self.gts = gts
        self.transforms = transform

    def __getitem__(self, index):
        # A_path = self.root_path + self.names_list_img[index]
        # B_path = self.root_path + self.names_list_gt[index]
        img = self.imgs[index]
        gt = self.gts[index]
        gt = gt.astype(int)
        # weights = 0
        img = self.transforms(image=img.copy(), mask=gt.copy())

        return img['image'], img['mask']
        # return {'A': img['image'], 'B': img['mask'], 'A_paths': A_path, 'B_paths': B_path, 'weights': weights}
    def __len__(self):
        return len(self.gts)

class train_dataset_noread_npc(Dataset):
    def __init__(self, imgs,gts,label_idx,transform):
        self.imgs = imgs
        self.gts = gts
        self.label_idx = label_idx
        self.transforms = transform

    def __getitem__(self, index):
        # A_path = self.root_path + self.names_list_img[index]
        # B_path = self.root_path + self.names_list_gt[index]
        img = self.imgs[index]
        gt = self.gts[index][self.label_idx]
        gt = gt.astype(int)
        # weights = 0
        print('a',img.shape, gt.shape,'\n')
        img = self.transforms(image=img.copy(), mask=gt.copy())

        return img['image'], img['mask']
        # return {'A': img['image'], 'B': img['mask'], 'A_paths': A_path, 'B_paths': B_path, 'weights': weights}
    def __len__(self):
        return len(self.gts)

class train_dataset_noread_distance(Dataset):
    def __init__(self, imgs,gts,transform):
        self.imgs = imgs
        self.gts = gts
        self.transforms = transform

    def __getitem__(self, index):
        # A_path = self.root_path + self.names_list_img[index]
        # B_path = self.root_path + self.names_list_gt[index]
        img = self.imgs[index]
        gt = self.gts[index]
        gt = gt.astype(int)
        #weights = 0

        img = self.transforms(image=img.copy(), mask=gt.copy())
        distance_debug = get_distance(img['mask'])
        return img['image'], img['mask'], torch.from_numpy(distance_debug)
        # return {'A': img['image'], 'B': img['mask'], 'A_paths': A_path, 'B_paths': B_path, 'weights': weights}
    def __len__(self):
        return len(self.gts)
class train_dataset_noread_distance_jit(Dataset):
    def __init__(self, imgs,gts,transform):
        self.imgs = imgs
        self.gts = gts
        self.transforms = transform
        s = 0.5
        self.jit = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

    def __getitem__(self, index):
        # A_path = self.root_path + self.names_list_img[index]
        # B_path = self.root_path + self.names_list_gt[index]
        img = self.imgs[index]
        gt = self.gts[index]
        gt = gt.astype(int)
        #weights = 0

        img = self.transforms(image=img.copy(), mask=gt.copy())
        distance_debug = get_distance(img['mask'])
        return img['image'],self.jit(img['image']), img['mask'], torch.from_numpy(distance_debug)
        # return {'A': img['image'], 'B': img['mask'], 'A_paths': A_path, 'B_paths': B_path, 'weights': weights}
    def __len__(self):
        return len(self.gts)

class train_dataset(Dataset):
    def __init__(self, path_img,path_gt,names, transform):
        self.path_img = path_img
        self.path_gt = path_gt
        self.names = names
        self.transforms = transform

    def __getitem__(self, index):
        img = cv2.imread(self.path_img + self.names[index])[:,:,::-1]#/255
        gt = cv2.imread(self.path_gt + self.names[index])[:,:,0]/255
        gt = gt.astype(int)
        # weights = 0
        img = self.transforms(image=img.copy(), mask=gt.copy())

        return img['image'], img['mask']

    def __len__(self):
        return len(self.names)
from torchvision import transforms


train_transform1 = A.Compose([
        A.RandomRotate90(),
        A.Flip(p=0.5),
        ], p=1.)
train_transform2 = A.Compose([
        A.ColorJitter(),
        A.GaussNoise(),
        ], p=1.)
train_transform3 = A.Compose([
        ToTensorV2()
        ], p=1.)

def get_distance(img,max_ds=10):

    distance = distance_transform_edt(img)
    distance2 = distance_transform_edt(1 - img)
    distance[distance > max_ds] = max_ds
    distance2[distance2 > max_ds] = max_ds
    final_distance = distance + distance2
    return final_distance
