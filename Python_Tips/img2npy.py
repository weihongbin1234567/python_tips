import os
import sys
import numpy as np
import cv2
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def img2npy():
    path = "/data/whb/Kvasir/Train/masks"
    save_path = '/data/whb/Kvasir/Train/img_npy'
    fileList = os.listdir(path)
    for file in fileList:
        img_path = os.path.join(path,file)
        npy_file = file[:-3] +'npy'
        npy_path = os.path.join(save_path,npy_file)
        mask  = cv2.imread(img_path, 0).astype(np.float32)
        mask  = cv2.resize( mask, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        np.save(npy_path,mask) 

# mask  = cv2.imread(path, 0).astype(np.float32)
# mask  = cv2.resize( mask, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
# np.save(save_path,mask)
def resize():
    path = "/data/whb/Kvasir/Train/images"
    save_path = '/data/whb/Kvasir/Train/img_npy'
    fileList = os.listdir(path)
    for file in fileList:
        img_path = os.path.join(path,file)
        print(img_path)
        save_file = os.path.join(save_path,file)
        img  = cv2.imread(img_path)
        img  = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(save_file, img)

# np.set_printoptions(threshold=np.inf) #完整显示矩阵数据
# print(mask)
# npy = np.load('mask.npy') #加载npy文件

if __name__ == '__main__':
    img2npy()
    # resize()





