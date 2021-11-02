# -*- coding: utf-8 -*-
import os
import cv2
import time
import numpy as np
import onnx
import onnxruntime

from config import MEAN, STD, COLORS, N_CLASS
from utils import load_img, dump_str, randering_mask


def test_onnx(test_dir, onnx_path, save_dir='onnx_output'):
    test_set = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')]
    test_num = len(test_set)
    print('test num:', test_num)
    os.makedirs(save_dir, exist_ok=True)

    session = onnxruntime.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    total_time = 0
    for i, path in enumerate(test_set):
        img_name = os.path.basename(path)
        print(f'{i+1}/{len(test_set)}, {img_name}')
        raw_img, input = load_img(path)
        t = time.time()
        inputs = {input_name: input}
        outs = session.run(None, inputs)
        output = outs[0].squeeze()
        pred_mask = np.argmax(output, axis=0)
        total_time += time.time() - t
        if True:
            img_draw = randering_mask(raw_img, pred_mask, N_CLASS, COLORS, alpha=0.8, beta=0.5)
            cv2.imwrite(os.path.join(save_dir, img_name[:-4] + '_cover.jpg'), img_draw)

    print('onnx:', total_time / len(test_set))


if __name__=="__main__":
    onnx_path = "model.onnx"
    test_dir = 'imgs_0.4_512'
    test_onnx(onnx_path, test_dir)
