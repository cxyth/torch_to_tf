# -*- coding: utf-8 -*-

import cv2
import numpy as np
from config import STD, MEAN


def load_img(path):
    ''' 读取数据及预处理示例 '''
    raw_img = cv2.imread(path)  # BGR
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB).astype(np.float32)   # RGB
    img = img / 255.    # 0. ~ 1.
    img = (img - MEAN) / STD    # 减均值，除方差
    img = np.transpose(img, (2, 0, 1))
    input_data = img[None, :, :, :].astype(np.float32)
    return raw_img, input_data


def dump_str(ndarray, txt_path):
    with open(txt_path, 'w') as f:
        for data in ndarray.flatten():
            f.writelines(str(data) + '\n')


def randering_mask(image, mask, n_label, colors, alpha=0.5, beta=0.5):
    '''
        渲染mask至image上
    :param image: 渲染的底图 (h*w*c)
    :type image: numpy
    :param mask: 所要渲染的二值图 (h*w)
    :type mask: numpy
    :param n_label: 标签种类数
    :type n_label: int
    :param colors: 颜色矩阵 exp:三个种类则[[255,0,255],[0,255,0],[255,0,0]]
    :type colors: numpy or list
    :return: opencv图像
    :rtype: opencv image
    '''
    colors = np.array(colors)
    mh, mw = mask.shape
    mask = np.eye(n_label)[mask.reshape(-1)]    # shape=(h*w, n_label),即长度为h*w的one-hot向量
    mask = np.matmul(mask, colors)  # (h*w,n_label) x (n_label,3) ——> (h*w,3)
    mask = mask.reshape((mh, mw, 3)).astype(np.uint8)
    return cv2.addWeighted(image, alpha, mask, beta, 0)

