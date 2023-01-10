
import os

import cv2
import mmcv
import numpy as np
from tqdm import tqdm

data_root_image = '/home/zxq/Desktop/MISdata/kvasir/Train/images'
data_root_mask = '/home/zxq/Desktop/MISdata/kvasir/Train/masks'
save_root_polyp = '/home/zxq/Desktop/MISdata/kvasir/Train/only_polyp'
save_root_back = '/home/zxq/Desktop/MISdata/kvasir/Train/only_back'

def get_polyp():
    image_name_list = sorted(os.listdir(data_root_image))
    for image_name in tqdm(image_name_list, total=len(image_name_list)):
        image_path = os.path.join(data_root_image, image_name)
        mask_path = os.path.join(data_root_mask, image_name)
        save_path = os.path.join(save_root_polyp, image_name)
        img = mmcv.imread(image_path)
        mask= mmcv.imread(mask_path)
        mask[mask <= 128] = 0
        mask[mask >= 128] = 255
        mask[mask <= 0.5] = 0
        mask[mask >= 0.5] = 1
        new_image = img*mask
        mmcv.imwrite(new_image, save_path,auto_mkdir=True)

def get_backgroud():
    image_name_list = sorted(os.listdir(data_root_image))
    for image_name in tqdm(image_name_list, total=len(image_name_list)):
        image_path = os.path.join(data_root_image, image_name)
        mask_path = os.path.join(data_root_mask, image_name)
        save_path = os.path.join(save_root_back, image_name)
        img = mmcv.imread(image_path)
        mask= mmcv.imread(mask_path)
        mask[mask <= 128] = 0
        mask[mask >= 128] = 128

        mask[mask <= 64] = 255
        mask[mask <= 200] = 0

        mask[mask <= 0.5] = 0
        mask[mask >= 0.5] = 1
        new_image = img*mask
        mmcv.imwrite(new_image, save_path,auto_mkdir=True)

if __name__ == '__main__':

    get_backgroud()
