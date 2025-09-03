import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import h5py,json
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
from skimage import exposure
from PIL import Image
class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        modality="t1c",
        transform=None
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.modality = modality

        if self.split == "train":
            self.sample_list = os.listdir(self._base_dir + "/training_2d")
        elif self.split == "val":
            self.sample_list = os.listdir(self._base_dir + "/validation")
        elif self.split == "test":
            self.sample_list = os.listdir(self._base_dir + "/testing")
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +"/training_2d/{}".format(case), "r")
        elif self.split == "val":
            h5f = h5py.File(self._base_dir + "/validation/{}".format(case), "r")
        elif self.split == "test":
            h5f = h5py.File(self._base_dir + "/testing/{}".format(case), "r")

        # image = h5f[self.modality][:]
        image_modality_list = ["t1", "t1c", "t2"]
        image = np.array([h5f[modality][:] for modality in image_modality_list])
        
        if self.split == "train":
            label = np.zeros((4, image.shape[1], image.shape[2]))
            label[0] = h5f["label_a1"][:]
            label[1] = h5f["label_a2"][:]
            label[2] = h5f["label_a3"][:]
            label[3] = h5f["label_a4"][:]
        else:
            label = np.zeros((4, image.shape[1], image.shape[2], image.shape[3]))
            label[0] = h5f["label_a1"][:]
            label[1] = h5f["label_a2"][:]
            label[2] = h5f["label_a3"][:]
            label[3] = h5f["label_a4"][:]
        # max_label= np.sum(label)
        sample = {"image": image, "label": label}
        sample = self.transform(sample)

        sample["idx"] = case
        return sample


class BaseDataSets_npc_distance_jit(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            modality="t1c",
            transform=None,
            tr_idx = 0    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.modality = modality
        self.tr_idx = tr_idx

        if self.split == "train":
            self.sample_list = os.listdir(self._base_dir + "/training_2d")
            self.sample_list.sort()

        elif self.split == "val":
            self.sample_list = os.listdir(self._base_dir + "/validation")
        elif self.split == "test":
            self.sample_list = os.listdir(self._base_dir + "/testing")
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +"/training_2d/{}".format(case), "r")
        elif self.split == "val":
            h5f = h5py.File(self._base_dir + "/validation/{}".format(case), "r")
        elif self.split == "test":
            h5f = h5py.File(self._base_dir + "/testing/{}".format(case), "r")

        # image = h5f[self.modality][:]
        image_modality_list = ["t1", "t1c", "t2"]
        image = np.array([h5f[modality][:] for modality in image_modality_list])

        if self.split == "train":
            label = np.zeros((4, image.shape[1], image.shape[2]))
            image_slic  = np.load('../data/MMIS2024TASK1/train_2D_slic/' + case.replace('.h5','.npy'))
            label[0] = h5f["label_a1"][:]
            label[1] = h5f["label_a2"][:]
            label[2] = h5f["label_a3"][:]
            label[3] = h5f["label_a4"][:]
            label = label[self.tr_idx]
        # max_label= np.sum(label)
        sample = {"image": image, "label": label,"slic": image_slic}
        sample = self.transform(sample)

        sample["idx"] = case
        return sample

class BaseDataSets_npc_distance_jit_select_one(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            modality="t1c",
            transform=None,
            tr_idx = 0 ,
        rate=1):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.modality = modality
        self.rate = rate
        # self.tr_idx = tr_idx

        if self.split == "train":
            self.sample_list = os.listdir(self._base_dir + "/training_2d")
            # self.sample_list.sort()
            random.seed(2025)
            random.shuffle(self.sample_list)
            self.sample_list = self.sample_list[:len(self.sample_list )//self.rate]

        elif self.split == "val":
            self.sample_list = os.listdir(self._base_dir + "/validation")
        elif self.split == "test":
            self.sample_list = os.listdir(self._base_dir + "/testing")
        print("total {} samples".format(len(self.sample_list)))
        with open("../data/MMIS2024TASK1/data_npc_select_one.json", "r") as f:
            self.loaded_dict = json.load(f)
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +"/training_2d/{}".format(case), "r")
        elif self.split == "val":
            h5f = h5py.File(self._base_dir + "/validation/{}".format(case), "r")
        elif self.split == "test":
            h5f = h5py.File(self._base_dir + "/testing/{}".format(case), "r")

        # image = h5f[self.modality][:]
        image_modality_list = ["t1", "t1c", "t2"]
        image = np.array([h5f[modality][:] for modality in image_modality_list])
        self.tr_idx = self.loaded_dict[case]
        if self.split == "train":
            label = np.zeros((4, image.shape[1], image.shape[2]))
            image_slic  = np.load('.//data/MMIS2024TASK1/train_2D_slic/' + case.replace('.h5','.npy'))
            label[0] = h5f["label_a1"][:]
            label[1] = h5f["label_a2"][:]
            label[2] = h5f["label_a3"][:]
            label[3] = h5f["label_a4"][:]
            label = label[self.tr_idx]
        # max_label= np.sum(label)
        sample = {"image": image, "label": label,"slic": image_slic}
        sample = self.transform(sample)

        sample["idx"] = case
        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)

    if len(image.shape) == 2:
        image = np.rot90(image, k)
        image = np.flip(image, axis=axis).copy()
    elif len(image.shape) == 3:
        for i in range(image.shape[0]):
            image[i] = np.rot90(image[i], k)
            image[i] = np.flip(image[i], axis=axis).copy()
    if len(label.shape) == 2:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
    elif len(label.shape) == 3:
        for i in range(label.shape[0]):
            label[i] = np.rot90(label[i], k)
            label[i] = np.flip(label[i], axis=axis).copy()

    return image, label

def random_rot_flip_our(image, label=None,slic=None):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)

    if len(image.shape) == 2:
        image = np.rot90(image, k)
        image = np.flip(image, axis=axis).copy()
        slic = np.rot90(slic, k)
        slic = np.flip(slic, axis=axis).copy()
    elif len(image.shape) == 3:
        for i in range(image.shape[0]):
            image[i] = np.rot90(image[i], k)
            image[i] = np.flip(image[i], axis=axis).copy()
    if len(label.shape) == 2:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
    elif len(label.shape) == 3:
        for i in range(label.shape[0]):
            label[i] = np.rot90(label[i], k)
            label[i] = np.flip(label[i], axis=axis).copy()

    return image, label,slic


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    if len(image.shape) == 2:
        image = ndimage.rotate(image, angle, order=0, reshape=False)
    elif len(image.shape) == 3:
        for i in range(image.shape[0]):
            image[i] = ndimage.rotate(image[i], angle, order=0, reshape=False)
    if len(label.shape) == 2:
        label = ndimage.rotate(label, angle, order=0, reshape=False)
    elif len(label.shape) == 3:
        for i in range(label.shape[0]):
            label[i] = ndimage.rotate(label[i], angle, order=0, reshape=False)
    return image, label
def random_rotate_our(image, label,slic):
    angle = np.random.randint(-20, 20)
    if len(image.shape) == 2:
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        slic = ndimage.rotate(slic, angle, order=0, reshape=False)
    elif len(image.shape) == 3:
        for i in range(image.shape[0]):
            image[i] = ndimage.rotate(image[i], angle, order=0, reshape=False)
    if len(label.shape) == 2:
        label = ndimage.rotate(label, angle, order=0, reshape=False)
    elif len(label.shape) == 3:
        for i in range(label.shape[0]):
            label[i] = ndimage.rotate(label[i], angle, order=0, reshape=False)
    return image, label,slic


def random_noise(image, label, mu=0, sigma=0.1):
    if len(image.shape) == 2:
        noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1]), -2 * sigma, 2 * sigma)
    elif len(image.shape) == 3:
        noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2 * sigma, 2 * sigma)
    else:
        pass
    noise = noise + mu
    image = image + noise
    return image, label


def random_rescale_intensity(image, label):
    image = exposure.rescale_intensity(image)
    return image, label

def random_equalize_hist(image, label):
    image = exposure.equalize_hist(image)
    return image, label

class RandomGenerator_Multi_Rater(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        _, x, y = image.shape

        image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
        if len(label.shape) == 2:
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        elif len(label.shape) == 3:
            label = zoom(label, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        if random.random() > 0.5:
            image, label = random_noise(image, label)
        # if random.random() > 0.5:
        #     image, label = random_rescale_intensity(image, label)
        # if random.random() > 0.5:
        #     image, label = random_equalize_hist(image, label)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample
from scipy.ndimage import distance_transform_edt
from torchvision import transforms
def get_distance(img,max_ds=10):

    distance = distance_transform_edt(img)
    distance2 = distance_transform_edt(1 - img)
    distance[distance > max_ds] = max_ds
    distance2[distance2 > max_ds] = max_ds
    final_distance = distance + distance2
    return final_distance
class RandomGenerator_Multi_Rater_our(object):
    def __init__(self, output_size):
        self.output_size = output_size
        s = 0.01
        self.jit = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    def __call__(self, sample):
        image, label,slic = sample["image"], sample["label"],sample["slic"]

        _, x, y = image.shape

        image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
        if len(label.shape) == 2:
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        elif len(label.shape) == 3:
            label = zoom(label, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)

        if random.random() > 0.5:
            image, label,slic = random_rot_flip_our(image, label,slic)
        if random.random() > 0.5:
            image, label,slic = random_rotate_our(image, label,slic)
        if random.random() > 0.5:
            image, label = random_noise(image, label)
        # if random.random() > 0.5:
        #     image, label = random_rescale_intensity(image, label)
        # if random.random() > 0.5:
        #     image, label = random_equalize_hist(image, label)

        image = torch.from_numpy(image.astype(np.float32))

        label = torch.from_numpy(label.astype(np.uint8))
        label_distance = torch.from_numpy(get_distance(label))
        slic = torch.from_numpy(slic.astype(np.uint8))
        sample = {"image": image,"image_jit":image, "label": label,"label_distance": label_distance,"slic": slic}
        return sample


class ZoomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        c, d, x, y = image.shape

        image = zoom(image, (1, 1, self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (1, 1, self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample
