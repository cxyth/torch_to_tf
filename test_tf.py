# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor, float32
from tensorflow import keras

from config import MEAN, STD, COLORS, N_CLASS
from utils import load_img, dump_str, randering_mask

'''
测试环境：
tensorflow-gpu      2.5.0
onnx                1.10.1
onnx2keras          0.0.24
'''

def test_h5(test_dir, h5_path, save_dir='h5_output'):
    test_set = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')]
    test_num = len(test_set)
    print('test num:', test_num)
    os.makedirs(save_dir, exist_ok=True)
    model = keras.models.load_model(h5_path)
    model.summary()
    total_time = 0
    for i, path in enumerate(test_set):
        img_name = os.path.basename(path)
        print(f'{i + 1}/{test_num}, {img_name}')
        raw_img, input = load_img(path)
        t = time.time()
        output = model.predict(input)
        output = np.squeeze(output)
        total_time += time.time() - t

        pred_mask = np.argmax(output, axis=0)
        cv2.imwrite(os.path.join(save_dir, img_name), pred_mask)
        if True:
            img_draw = randering_mask(raw_img, pred_mask, N_CLASS, COLORS, alpha=0.8, beta=0.5)
            cv2.imwrite(os.path.join(save_dir, img_name[:-4] + '_cover.jpg'), img_draw)
    print('h5:', total_time / len(test_set))


def test_pb(test_dir, pb_path, save_dir='pb_output'):
    test_set = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')]
    test_num = len(test_set)
    print('test num:', test_num)
    os.makedirs(save_dir, exist_ok=True)
    model = tf.saved_model.load(pb_path)
    total_time = 0
    for i, path in enumerate(test_set):
        img_name = os.path.basename(path)
        print(f'{i+1}/{test_num}, {img_name}')
        raw_img, input = load_img(path)
        t = time.time()
        # input = convert_to_tensor(input, dtype=float32)
        output = model(input)
        output = np.squeeze(output)
        total_time += time.time() - t

        # dump_str(output, os.path.join(save_dir, img_name[:-4] + '.txt'))

        pred_mask = np.argmax(output, axis=0)
        cv2.imwrite(os.path.join(save_dir, img_name), pred_mask)
        if True:    # 可视化渲染
            img_draw = randering_mask(raw_img, pred_mask, N_CLASS, COLORS, alpha=0.8, beta=0.5)
            cv2.imwrite(os.path.join(save_dir, img_name[:-4] + '_cover.jpg'), img_draw)
    print('done! {:.2f}s'.format(total_time))


if __name__=="__main__":
    pb_path = 'tf_model'
    test_dir = 'imgs_0.4_512'
    test_pb(test_dir, pb_path)
