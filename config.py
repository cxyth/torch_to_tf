# -*- coding: utf-8 -*-


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

COLORS = [
    [0, 0, 0],  # background,
    [255, 0, 255], # building
    [0, 0, 255],  # road,
    [255, 255, 0], # farmland
    [255, 0, 0],  # water,
    [0, 255, 0], # forest
    [0, 128, 0], # glass
    [0, 255, 255], # bare1
    [0, 128, 128] # bare2
]

N_CLASS = len(COLORS)

