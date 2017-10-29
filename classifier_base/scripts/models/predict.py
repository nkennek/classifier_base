#!/usr/bin/env python
# -- coding:utf-8 --

import argparse
import os
import pickle
import re
import sys
import time
from datetime import datetime as dt

import chainer
import chainer.functions as F
import numpy as np
import pandas as pd
from chainer import cuda
from chainer import serializers
from skimage import io

from classifier_base.scripts.models.data_model import PreprocessedDataset
from classifier_base.scripts.models.model_config import ModelConfig


MAX_IMG_ON_MEMORY = 1000


class Predictor(object):
    def __init__(self, cfg, model_path, gpu=False):
        self.config = cfg
        self.model = self._build_model(cfg, model_path, use_gpu=gpu)
        self.use_gpu = gpu

    def __call__(self, img):
        """
        predict single image
        """
        with chainer.using_config('train', False):
            img = PreprocessedDataset.preprocess(img, self.config.input_size, self.config.native_size, horizontal_flip=False, test=True)
            img = np.array([img])
            if self.use_gpu:
                img = cuda.to_gpu(img, device=self.config.gpu_id)
            return self.model.infer(img).data[0]

    def batch_predict(self, imgs):
        """
        predict multiple images
        """
        image_num = len(imgs)
        batch_size = self.config.test_batch_size
        batch_num = image_num // self.config.test_batch_size + 1
        pred_all = np.empty((0, self.config.num_classes))
        with chainer.using_config('train', False):
            imgs = np.array([
                PreprocessedDataset.preprocess(img, self.config.input_size, self.config.native_size, horizontal_flip=False, test=True)
                for img in imgs
            ])
            for i in range(batch_num):
                print('{}: read & predict {}/{}'.format(dt.now(), batch_size*i, image_num), end='\r', flush=True)
                idx_start = batch_size*i
                if idx_start >= image_num:
                    break
                idx_end = min(batch_size*(i+1), image_num)
                imgs_batch = imgs[idx_start:idx_end]
                if self.use_gpu:
                    imgs_batch = cuda.to_gpu(imgs_batch, device=self.config.gpu_id)
                pred = self.model.infer(imgs_batch).data
                pred_all = np.append(pred_all, pred, axis=0)
        return pred_all

    def predict_local_files(self, img_paths):
        """
        img_paths: file paths to images
        """
        image_num = len(img_paths)
        if image_num < MAX_IMG_ON_MEMORY:
            imgs = [io.imread(img_path, 0) for img_path in img_paths]
            return self.batch_predict(imgs)

        batch_num = image_num // MAX_IMG_ON_MEMORY + 1
        pred_all = np.empty((0, self.config.num_classes))
        for i in range(batch_num):
            start = i*MAX_IMG_ON_MEMORY
            end   = (i+1)*MAX_IMG_ON_MEMORY
            imgs = [io.imread(img_path, 0) for img_path in img_path[start:end]]
            pred =  self.batch_predict(imgs)
            pred_all = np.append(pred_all, pred, axis=0)

        return pred_all

    def _build_model(self, cfg, model_path, use_gpu=False):
        #load model
        if model_path == 'newest':
            model_path = get_newest_modelpath(cfg)
        print('{}: start to load {}'.format(dt.now(), model_path))
        model, _ = cfg.setup(use_gpu=use_gpu)
        serializers.load_hdf5(model_path, model)
        return model


def get_newest_modelpath(cfg):
    newest_date = None
    newest_model = None
    for model in os.listdir(cfg.trained_model_path):
        if model.split('.')[-1] != 'h5':
            continue
        date = match_date(model)
        if newest_date is None or newest_date < date:
            newest_date = date
            newest_model = os.path.join(cfg.trained_model_path, model)
    if newest_model is None:
        raise FileNotFoundError('There seems no model file')
    print(newest_model)
    return newest_model


def match_date(model_name):
    model_name, _ = os.path.splitext(model_name)
    try:
        date_str = model_name.split('_')[-1]
        date_dt = dt.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
    except:
        sys.stderr.write(f'warning: failed to parse date in model name {model_name}\n')
        # represents no match with oldest date
        date_dt = dt.min
    return date_dt


def execute_predict(config, model_path=None, file_provided=None, dir_provided=None, out=None, gpu=False):
    #load model
    if model_path == 'newest' or model_path is None:
        model_path = get_newest_modelpath(config)

    # use the same config as the one in training, if accesible
    model_name, _ = os.path.splitext(os.path.basename(model_path))
    config_path = os.path.join(config.log_path, model_name, 'config.dump')
    if os.path.isfile(config_path):
        with open(config_path, 'rb') as f:
            config = pickle.load(f)

    # set input&output directory
    if out is None:
        out = os.path.join(config.log_path, model_name)
    if dir_provided is not None:
        config.to_predict_data_path = dir_provided

    #load images
    image_names = []
    if file_provided is not None:
        image_names.append(file_provided)
    else:
        for f in os.listdir(config.to_predict_data_path):
            image_names.append(os.path.join(config.to_predict_data_path, f))

    start_time = dt.now()
    clf = Predictor(config, model_path, gpu=False)
    pred_all = clf.predict_local_files(image_names)
    time_spent = (dt.now() - start_time).total_seconds()
    time_per_img = time_spent/len(image_names)

    print('{}: finished'.format(dt.now()))
    print(f'''time spent: \n
        {time_spent} [s] in total\n
        {time_per_img} [s] per images''')

    filename = 'pred_{}.csv'.format(dt.now())
    pd.DataFrame(pred_all, index=image_names).to_csv(os.path.join(out, filename), header=False)
    print(f'result is saved at {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict whether image represents private room or not.")
    parser.add_argument('--model', action='store', nargs='?', help='path to model for judge. if not given, most recent one will be automatically used')
    parser.add_argument('--file', action='store', nargs='?', help='if provided, an image is loaded')
    parser.add_argument('--dir', action='store', nargs='?', help='if provided, image are loaded from fed directory default: ../../data/to_predict')
    parser.add_argument('--out', action='store', nargs='?', help='path to output result.')
    args = parser.parse_args()

    config = ModelConfig(base_dir=os.path.dirname(__file__))
    execute_predict(config, args.model, args.file, args.dir, args.out)
