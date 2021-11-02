# -*- coding: utf-8 -*-
import os
import cv2
import time
import numpy as np
import onnx
import torch
import torchvision

from config import MEAN, STD, COLORS, N_CLASS

'''
测试环境：
pytorch             1.7.1
onnx                1.10.1
'''


def create_torch_model(arch, encoder, in_channel, out_channel, pretrained=None):
    import segmentation_models_pytorch as smp
    smp_net = getattr(smp, arch)
    model = smp_net(               # smp.UnetPlusPlus
        encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=pretrained,     # use `imagenet` pretrained weights for encoder initialization
        in_channels=in_channel,     # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=out_channel,     # model output channels (number of classes in your dataset)
    )
    return model


def export_alexnet_onnx(out_path):
    # Standard ImageNet input - 3 channels, 224x224,
    # values don't matter as we care about network structure.
    # But they can also be real inputs.
    dummy_input = torch.randn(1, 3, 224, 224)
    # Obtain your model, it can be also constructed in your script explicitly
    model = torchvision.models.alexnet(pretrained=True)
    # Invoke export
    torch.onnx.export(model, dummy_input, out_path)


def print_onnx(inpath):
    # Load the ONNX model
    model = onnx.load(inpath)
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))


def onnx_simplify(onnx_in, onnx_out):
    path, model = os.path.split(onnx_in)
    model_name, _ = os.path.splitext(model)
    # load your predefined ONNX model
    model = onnx.load(onnx_in)
    # convert model
    model_simp, check = simplify(model)
    onnx.save(model_simp, onnx_out)
    if check:
        print("Ok!")
    else:
        print(
            "Check failed, please be careful to use the simplified model, or try specifying \"--skip-fuse-bn\" or \"--skip-optimization\" (run \"python3 -m onnxsim -h\" for details)")
        exit(1)


if __name__=="__main__":
    onnx_path = "model.onnx"
    ckpt_file = 'upp_rsn50.v8.1/ckpt/checkpoint-epoch92.pth'

    checkpoint = torch.load(ckpt_file)
    state_dict = checkpoint['state_dict']
    model = create_torch_model(
        arch='UnetPlusPlus',
        encoder='resnet50',
        in_channel=3,
        out_channel=N_CLASS).cuda()
    model.load_state_dict(state_dict)

    dummy_input = torch.randn(1, 3, 512, 512)
    torch.onnx.export(model, dummy_input, onnx_path)
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))

