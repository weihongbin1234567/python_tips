import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt

def npz2img():
    sample_path = "./sample/samples_20x256x256x3.npz"
    save_path = "./sample_2k_DDPM_m_10w_images"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print('make dir')
    npzfile = np.load(sample_path)
    sample = npzfile['arr_0']
    for i in range(sample.shape[0]):
        print(i)
        img = (sample[i]).astype(np.uint8)
        img = Image.fromarray(img,mode='RGB')
        save_img_path = os.path.join(save_path, f"img_{i}.png")
        img.save(save_img_path)
    print("sample end")

def img_joint():
    fig = plt.figure()
    for i in range(16):

        path = os.path.join('./sample_2k_DDPM_m_10w_images', f'img_{i}.png')
        img = Image.open(path)
        point = i + 1 
        plt.subplot(4,4,point)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig("fig.png", bbox_inches='tight')

    # img1 = Image.open("./sample_images/img_0.png")
    # img2 = Image.open("./sample_images/img_1.png")
    # plt.subplot(2,4,1)
    # plt.imshow(img1)
    # plt.axis('off')
    # plt.subplot(2,4,2)
    # plt.imshow(img2)
    # plt.axis('off')
    # plt.savefig("fig.png", bbox_inches='tight')

    print("pol end")

if __name__ == "__main__":
    npz2img()
    img_joint()




