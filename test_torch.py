# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import time
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import MEAN, STD, COLORS, N_CLASS
from utils import load_img, dump_str, randering_mask


def get_val_transform(input_size):
    return A.Compose([
                A.Resize(input_size, input_size),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ])


def create_model(arch, encoder, in_channel, out_channel, pretrained=None):
    smp_net = getattr(smp, arch)
    model = smp_net(               # smp.UnetPlusPlus
        encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=pretrained,     # use `imagenet` pretrained weights for encoder initialization
        in_channels=in_channel,     # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=out_channel,     # model output channels (number of classes in your dataset)
    )
    return model


def test_pytorch(test_dir, model, save_dir='torch_output'):
    test_set = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')]
    test_num = len(test_set)
    print('test num:', test_num)
    os.makedirs(save_dir, exist_ok=True)
    transform = get_val_transform(256)
    total_time = 0
    model.eval()
    for i, path in enumerate(test_set):
        img_name = os.path.basename(path)
        print(f'{i + 1}/{test_num}, {img_name}')
        raw_img = cv2.imread(path)
        _img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        _img = transform(image=_img)['image']
        _img = _img.unsqueeze(0)
        t = time.time()
        with torch.no_grad():
            _img = _img.cuda()
            output = model(_img)
            output = output.squeeze().cpu().numpy()
        total_time += time.time() - t
        pred_mask = np.argmax(output, axis=0)

        if True:
            img_draw = randering_mask(raw_img, pred_mask, N_CLASS, COLORS, alpha=0.8, beta=0.5)
            cv2.imwrite(os.path.join(save_dir, img_name[:-4] + '_cover.jpg'), img_draw)
    print('pytorch:', total_time / len(test_set))


if __name__=="__main__":
    test_dir = 'imgs_0.4_512'
    ckpt_file = 'upp_rsn50.v8.1/ckpt/checkpoint-epoch92.pth'

    checkpoint = torch.load(ckpt_file)
    state_dict = checkpoint['state_dict']
    model = create_model(arch='UnetPlusPlus',
                         encoder='resnet50',
                         in_channel=3,
                         out_channel=N_CLASS).cuda()
    model.load_state_dict(state_dict)

    test_pytorch(test_dir, model)
