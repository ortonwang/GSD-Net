
import os
from  tqdm import tqdm
import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float

path_img_dir = '../data/Kvasir/images/'
save_dir = '../data/Kvasir/images_SLIC/'
os.makedirs(save_dir, exist_ok=True)
names = os.listdir(path_img_dir)
for name in tqdm(names):
    image = cv2.imread(path_img_dir + name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB
    image = img_as_float(image)  #
    segments = slic(image, n_segments=1024, compactness=10, sigma=1, start_label=0)  # 默认是1024
    np.save(save_dir + name.replace('.png','.npy').replace('.jpg','.npy'), segments)

