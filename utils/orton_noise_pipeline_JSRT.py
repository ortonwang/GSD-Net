
import skimage.io
import cv2
from glob import glob
import numpy as np
import json
import os
import shutil

import ipdb
import random
from noise_generation import add_noise

path_clean = '//mnt/orton/dataset/1_data_for_label_noise/shenzhen/mask/'
result_dir = '//mnt/orton/dataset/1_data_for_label_noise/shenzhen/shenzhen_1_07/'
os.makedirs(result_dir,exist_ok=True)
names = os.listdir(path_clean)
for name in names:
    print(name)
    clean_label_path = path_clean + name
    noisy_label_path = result_dir + name
    clean_label = skimage.io.imread(clean_label_path, as_gray=True)
    clean_label[clean_label > 0] = 1
    noisy_label, noise_type = add_noise(clean_label, noise_ratio=0.7)
    noisy_label[noisy_label > 0] = 255  # for visualize
    noisy_label = noisy_label.astype(np.uint8)
    skimage.io.imsave(noisy_label_path, noisy_label, check_contrast=False)
