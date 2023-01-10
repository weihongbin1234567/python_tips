
#jpg图片转png
import PIL.Image
import os
i=0
path = "/home/zxq/Desktop/Net/dataset/Test/video/images/"
savepath = "/home/zxq/Desktop/Net/dataset/Test/video/image/"
filelist = os.listdir(path)
for file in filelist:
    im = PIL.Image.open(path+filelist[i])
    filename = os.path.splitext(file)[0]
    im.save(savepath+filename+'.png') # or 'test.tif'
    i=i+1


