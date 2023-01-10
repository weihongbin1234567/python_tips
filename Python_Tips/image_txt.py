import os

def createFileList(images_path, txt_save_path):
    # 打开图片列表清单txt文件
    fw = open(txt_save_path, "w")
    # 查看文件夹下的图片名称
    images_name = os.listdir(images_path)
    count = 0
    # 遍历所有文件名
    for eachname in images_name:
        # 按照规则将内容写入txt文件中
        # name = os.path.splitext(eachname)
        # print(name)
        #filename = name[0]  
        ##这里更改的图片名称Newdir：05_001.jpg,可更改后面的“.jpg”来改变读取的图片格式，如“png”
        fw.write(eachname +'\n')
        #将读取的图片名称与更改后的图片名称，同时保存在txt文件中
        count += 1
    # 打印成功信息
    print("生成txt文件成功")
    # 关闭fw
    fw.close()


# 下面是相关变量定义的路径
if __name__ == '__main__':
   

    # 图片存放目录 
    images_path = '/home/zxq/Desktop/MIDdata/kvasir/Train/FID/label_data/label'
    # 生成的图片列表清单txt文件的保存目录
    txt_save_path = "/home/zxq/Desktop/MIDdata/kvasir/Train/FID/label_data/train_full_list.txt" 

    createFileList(images_path, txt_save_path)
