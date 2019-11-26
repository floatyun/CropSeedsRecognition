import os
import cv2 as cv2
import numpy as np
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import seaborn as sns
from judge import show_heat_img
# from scipy import misc


def Gener_mat(a, b, x, y, w, h):  # 生成图片矩阵
    img_mat = np.zeros((a, b), dtype=np.int)
    for i in range(0, a):
        for j in range(0, b):
            img_mat[i][j] = 0
    for i in range(x, x + w):
        for j in range(y, y + h):
            img_mat[i][j] = 1
    return img_mat


def out_img(data):  # 输出图片
    new_im = Image.fromarray(data)  # 调用Image库，数组归一化
    # new_im.show()
    plt.imshow(data)  # 显示新图片


def get_heat_img(mtr, filename="heat_img.jpg"):
    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(mtr, annot=True, ax=ax)  # 画热力图
    ax.set_title('cluster result heat image')  # 标题
    ax.set_xlabel('real class')  # x轴
    ax.set_ylabel('cluster class')  # y轴
    f.savefig(filename, bbox_inches='tight')


def mtr_to_img(mtr,filename):
    # print(mtr.shape)
    k = 240
    display_mtr = np.zeros([mtr.shape[0]*k, mtr.shape[1]*k], dtype=np.uint8)
    display_mtr = np.zeros([mtr.shape[0] * k, mtr.shape[1] * k])
    for i in range(display_mtr.shape[0]):
        for j in range(display_mtr.shape[1]):
            display_mtr[i, j] = np.uint8(mtr[i//k, j//k] * 255)
            # display_mtr[i, j] = mtr[i//k, j//k]
    get_heat_img(mtr,filename)
    show_heat_img(filename)
    # im_color = cv2.applyColorMap(display_mtr, cv2.COLORMAP_JET)
    # cv2.imwrite(filename,display_mtr)
    # cv2.imshow("GrayImage", display_mtr)
    # cv2.waitKey()
    # print(type(mtr))
    # im = Image.fromarray(mtr)
    # im.save(filename)
    # misc.imsave(filename, display_mtr)
    #  使用PIL
    # im = Image.fromarray(display_mtr)
    # im.save(filename)
