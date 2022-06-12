import os
import random

import cv2
import numpy as np
import imutils
# D:\Desktop\yolo\out\more_img/
def sp_noise(noise_img, proportion):
    height, width = noise_img.shape[0], noise_img.shape[1]#获取高度宽度像素值
    num = int(height * width * proportion) #一个准备加入多少噪声小点
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    return noise_img


def gaussian_noise(img, mean, sigma):
    # 将图片灰度标准化
    img = img / 255
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out*255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out# 这里也会返回噪声，注意返回值


def random_noise(image, noise_num):
    # cv2.imshow("src", img)
    rows, cols, chn = image.shape
    # 加噪声
    for i in range(noise_num):
        x = np.random.randint(0, rows)#随机生成指定范围的整数
        y = np.random.randint(0, cols)
        image[x-1:x+1, y-1:y+1, :] = 255
    return image

path = r'big'
save_path = r'D:\Desktop\yolo\out\more_img/'
path_list = os.listdir(path)
for i in path_list:
    img_path = path+'/'+i
    img = cv2.imread(img_path)
    rot1 = gaussian_noise(img, 0, 0.15)  # 高斯噪声
    rot2 = random_noise(img, 100)
    rot3 = sp_noise(img, 0.025)
    # rot1 = imutils.rotate_bound(img, 5)
    # rot2 = imutils.rotate_bound(img, -5)
    # rot3 = imutils.rotate_bound(img, 15)
    # rot4 = imutils.rotate_bound(img, -15)
    img1 = cv2.resize(rot1, (128, 128))
    img2 = cv2.resize(rot2, (128, 128))
    img3 = cv2.resize(rot3, (128, 128))
    # img4 = cv2.resize(rot4, (128, 128))
    # cv2.imwrite(save_path + i.split('.')[0] + 'fl.jpg', img_re)
    cv2.imwrite(save_path + i.split('.')[0] + 'gn.jpg', img1)
    # cv2.imwrite(save_path + 'kai_' + i, result)
    # cv2.imwrite(save_path + 'gs_' + i, img_gs)
    cv2.imwrite(save_path + i.split('.')[0] + 'rn.jpg', img2)
    cv2.imwrite(save_path + i.split('.')[0] + 'sn.jpg', img3)
    # cv2.imwrite(save_path + i.split('.')[0] + 'r165.jpg', img4)
    print(i + ' have saved more img')
    # cv2.waitKey(0)
