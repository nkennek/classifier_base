#!/usr/bin/env python
# -- coding:utf-8 --

import cv2
import numpy as np
from scipy.misc import imresize


def resize_image(img, minimum_length=256):
    """
    resize image while keeping aspect
    """
    y, x = img.shape[:2]
    # keep aspect ratio
    if y <= x:
        scale = minimum_length / y
        sizes = (minimum_length, int(scale * x))
    else:
        scale = minimum_length / x
        sizes = (int(scale * y), minimum_length)
    # If grey picture
    if img.ndim == 2:
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
    return imresize(img, sizes, interp='bilinear', mode='RGB')


def crop_center(img, sizes=(224, 224)):
    y, x, channel = img.shape
    center_y, center_x = int(y / 2), int(x / 2)
    frame_y, frame_x = sizes
    up, down = -int((frame_y + 1) / 2), int(frame_y / 2)
    left, right = -int((frame_x + 1) / 2), int(frame_x / 2)
    start_h, end_h = max(center_y + up, 0), min(center_y + down, y)
    start_w, end_w = max(center_x + left, 0), min(center_x + right, x)
    return img[start_h:end_h, start_w:end_w]


def crop_randomly(img, sizes=(224, 224)):
    """
    scale augmentation
    used in ImageNet classification on ResNet(https://arxiv.org/abs/1512.03385) 
    """
    y, x, channel = img.shape
    length_y, length_x = sizes
    # pick random number
    keypoint_y = np.random.randint(1, y - length_y + 1)
    keypoint_x = np.random.randint(1, x - length_x + 1)
    start_y = keypoint_y
    end_y = keypoint_y + length_y
    start_x = keypoint_x
    end_x = keypoint_x + length_x
    return img[start_y: end_y, start_x: end_x]


def erase_randomly(img, p=0.5, sl=0.02, sh=0.4, r1=1/3., r2=3.):
    """
    random erasing(https://arxiv.org/abs/1708.04896)

    p     : Random Erasingを使用する確率
    sl,sh : マスクする領域の最小比率と最大比率(画像全体に対する面積比)
    r1,r2 : マスク領域のアスペクト比の最小値と最大値
    """

    if p > np.random.rand():
        img_width, img_height = img.shape[:2]
        mask_aspect = np.random.uniform(r1, r2)
        mask_width = np.random.uniform(sl, sh) * img_width
        mask_height = np.random.uniform(sl, sh) * img_height
        if mask_height / mask_width < r1:
            mask_height = mask_width*r1
        elif mask_height / mask_width > r2:
            mask_height = mask_width*r2
        mask_width, mask_height = int(mask_width), int(mask_height)

        center_x, center_y = np.random.randint(int(mask_width/2), img_width - int(mask_width/2)), np.random.randint(int(mask_height/2), img_height - int(mask_height/2))
        erased_color = np.random.randint(-128, 128)
        img[center_x-int(mask_width/2):center_x+int(mask_width/2),center_y-int(mask_height/2):center_y+int(mask_height/2), :] = erased_color

    return img
