# !/usr/bin/python
# -*- coding: UTF-8 -*-
import os, sys
# import cv2
from PIL import Image
old_path = r'C:\Users\weihongbin1\Desktop\dataset\PolypDataset\Test'  
dirs = os.listdir(old_path)
new_path =  r'C:\Users\weihongbin1\Desktop\dataset\polyp' 
count = 0
# os.path.join()
for dir in dirs:
    dir_path = os.path.join(old_path,dir)
    # print(path)
    dir_path = os.path.join(dir_path,"images")
    fileList = os.listdir(dir_path)
    for img in fileList:
        img_path = os.path.join(dir_path,img)
        print(img_path)
        img = Image.open(img_path)
         # 设置新文件名
        newname =  os.path.join(new_path, 'test' + str(count) +'.png')  
        count += 1
        img.save(newname)

print('重命名文件总数：', count)



