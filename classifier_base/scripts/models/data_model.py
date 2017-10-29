#!/usr/bin/env python
# -- coding:utf-8 --

import sys

import cv2
import chainer
import numpy as np
from skimage import io
from classifier_base.scripts.models.processing_func import *


class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, x, y, crop_size=(224, 224), resize=(256, 512), horizontal_flip=True, test=False, gpu=-1):
        self.x, self.y = x, y
        self.crop_size = crop_size
        self.resize = resize
        self.horizontal_flip = horizontal_flip
        self.test = test
        self.gpu = gpu

    def __len__(self):
        return len(self.x)

    def get_example(self, i):
        # Load image
        img = io.imread(self.x[i], 0)
        img = self.preprocess(img, self.crop_size, self.resize, self.horizontal_flip, self.test)
        # Label
        t = np.array(self.y[i], dtype=np.int32)
        return img, t

    @staticmethod
    def preprocess(img, crop_size, resize, horizontal_flip=True, test=False):
        # Resize image in the range
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.cv2.COLOR_RGBA2RGB)
        if img.ndim == 1:
            #read PIL imageplugin along
            img = img[0]

        img = resize_image(img, minimum_length=int(np.random.randint(*resize)))
        if test:
            # Crop center for test
            img = crop_center(img, sizes=crop_size)
        else:
            # Crop randomly
            img = crop_randomly(img, sizes=crop_size)
            if horizontal_flip and np.random.rand() >= 0.5:
                # Horizontal filp with 0.5
                img = cv2.flip(img, 1)
            img = erase_randomly(img)
        # To BGR
        img = img[:, :, ::-1]
        # Subtract imagenet mean
        img = np.array(img, dtype=np.float32) - np.array([103.063,  115.903,  123.152], dtype=np.float32)
        img = img / 255.
        # (channel , height, width)
        img = img.transpose((2, 0, 1))
        return img
