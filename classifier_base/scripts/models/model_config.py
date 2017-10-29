#!/usr/bin/env python
# -- coding:utf-8 --

import os
import pickle

import chainer

from classifier_base.scripts.models.networks import *

DIRNAME = os.path.dirname(os.path.abspath(__file__))

class ModelConfig(object):
    """
    学習用の定数や使用するモデル等のメタ情報
    """

    def __init__(self, base_dir=None):
        #ディレクトリ構成
        if base_dir is None:
            base_dir = DIRNAME

        self.raw_data_path           = os.path.normpath(os.path.join(base_dir, '../../data/full'))
        self.processed_data_path     = os.path.normpath(os.path.join(base_dir, '../../data/processed'))
        self.to_predict_data_path    = os.path.normpath(os.path.join(base_dir, '../../data/to_predict'))
        self.trained_model_path      = os.path.normpath(os.path.join(base_dir, '../../models/trained'))
        self.pretrained_model_path   = os.path.normpath(os.path.join(base_dir, '../../models/pretrained'))
        self.log_path                = os.path.normpath(os.path.join(base_dir, '../../log'))

        for path in [self.raw_data_path, self.processed_data_path, self.trained_model_path, self.pretrained_model_path, self.log_path]:
            if not os.path.exists(path):
                os.makedirs(path)

        #ハイパーパラメータや学習のための定数の設定
        self.random_state = 1234
        self.val_size = 0.1
        self.input_size = (224, 224)
        self.native_size = (256, 257)
        self.category = os.listdir(self.raw_data_path)
        self.num_classes = len(self.category)
        self.max_epoch = 20
        self.batch_size = 10
        self.learning_rate = 1.e-3
        self.momentum = 0.9
        self.weight_decay_rate = 5.e-4
        self.lr_decay_interval = 5, 'epoch'

        #その他
        self.val_interval = 100, 'iteration'
        self.log_interval = 100, 'iteration'
        self.test_batch_size = 10
        self.gpu_id = 0
        self.model_info = "FineTuneResNet(os.path.join(self.pretrained_model_path, 'resnet_50.caffemodel'), 50, self.num_classes)"

    #setup model and optimizer here
    def setup(self, use_gpu=True):
        print(self.pretrained_model_path)
        self.model = FineTuneResNet(os.path.join(self.pretrained_model_path, 'resnet_50.caffemodel'), 50, self.num_classes)
        if use_gpu:
            self.model.to_gpu(self.gpu_id)
        self.optimizer = chainer.optimizers.MomentumSGD(lr=self.learning_rate, momentum=self.momentum)
        self.optimizer.setup(self.model)
        self.weight_decay = chainer.optimizer.WeightDecay(self.weight_decay_rate)
        self.optimizer.add_hook(self.weight_decay)
        return self.model, self.optimizer

    def dump(self, exec_id):
        pickle_f = os.path.join(self.log_path, exec_id, 'config.dump')
        with open(pickle_f, 'wb') as f:
            pickle.dump(self, f)

    #for make it picklable
    def __getstate__(self):
        state = self.__dict__.copy()
        # delete unpicklable properties
        for unpicklable_prop in ['model', 'optimizer', 'weight_decay']:
            if unpicklable_prop in dir(self):
                del state[unpicklable_prop]
        return state

    # for make it from pickle
    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        state = self.__getstate__()
        print_str = ''
        for attr, value in state.items():
            print_str = print_str + f'{attr}: {value}\n'

        return print_str


if __name__ == '__main__':
    """
    学習に必要なディレクトリを構成 & 定数の確認
    """
    config = ModelConfig()
    print(config)
