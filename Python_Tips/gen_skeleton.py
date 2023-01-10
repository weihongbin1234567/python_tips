# -*- coding: utf-8 -*-
# @Time    : 2020/9/26
# @Author  : Lart Pang
# @Email   : lartpang@163.com
# @File    : gen_skeleton.py
# @Project : USVideoSeg
# @GitHub  : https://github.com/lartpang
import os

import cv2
import mmcv
import numpy as np
from skimage import morphology,draw
from tqdm import tqdm

data_root = '/home/zxq/Downloads/datasets/miccai_segmentation/TrainDataset/TrainDataset/mask'
save_root = '/home/zxq/Downloads/datasets/miccai_segmentation/TrainDataset/TrainDataset/skeleton'



def main():
    image_name_list = sorted(os.listdir(data_root))
    for image_name in tqdm(image_name_list, total=len(image_name_list)):
        image_path = os.path.join(data_root, image_name)
        save_path = os.path.join(save_root, image_name)
        img = cv2.imread(image_path, 0)
        # print(img.max())
        # _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        # cv2.imwrite("binary.png", binary)
        img[img <128] = 0
        img[img !=0] = 1
        skeleton0 = morphology.skeletonize(img)
        skeleton = skeleton0.astype(np.uint8) * 255
        cv2.imwrite(save_path, skeleton)



if __name__ == '__main__':
    # mask = np.zeros(shape=(1000, 1000))
    # mask[200:700, 400:900] = 1
    # bbox = get_bbox(mask=mask)
    # get_hotspot(mask=mask, bbox=bbox, )
    main()
