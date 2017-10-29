#!/usr/bin/env python
# -- coding:utf-8 --

import os
import sys
from datetime import datetime as dt

import chainer
import numpy as np
import pandas as pd
from chainer import serializers
from chainer import training
from chainer.training import extensions
from sklearn.model_selection import train_test_split

#for PlotReport on server
import matplotlib as mpl
mpl.use('Agg')

from classifier_base.scripts.models.data_model import PreprocessedDataset
from classifier_base.scripts.models.model_config import ModelConfig


def execute_train(config):
    timestamp = dt.now().strftime('%Y-%m-%d_%H:%M')
    model, optimizer = config.setup()
    exec_id = "Training_{}_{}".format(model.__class__.__name__, timestamp)

    #prepare datasets
    filenames, labels = fetch_filename_and_labels(config)
    train_X, test_X, train_y, test_y = train_test_split(
        filenames, labels, test_size=config.val_size, random_state=config.random_state
        )
    try:
        model, val = train(config, model, optimizer, train_X, test_X, train_y, test_y, exec_id)
    except KeyboardInterrupt:
        save_model(config, "{}.h5".format(exec_id), model)
        save_data_info(config, exec_id, train_X, test_X, train_y, test_y) # log which data is used for train & val
        config.dump(exec_id)
        sys.exit(0)

    save_model(config, "{}.h5".format(exec_id), model)
    save_data_info(config, exec_id, train_X, test_X, train_y, test_y) # log which data is used for train & val
    config.dump(exec_id)

    # validation result
    print('all informations are saved. computing validation result in detail...')
    from sklearn.metrics import classification_report
    with chainer.using_config('train', False):
        model.to_cpu()
        test_set = val[np.arange(len(test_X))]
        imgs, labels = np.array([t[0] for t in test_set]), np.array([t[1] for t in test_set])
        pred_labels = np.argmax(model.infer(imgs).data, axis=1)
        print(classification_report(labels, pred_labels, target_names=config.category))


def train(config, model, optimizer, train_X, test_X, train_y, test_y, exec_id):
    '''
    Train model with optimizer
        - (train_X, train_y) are used for update
        - (test_X, test_y) are used for validation
        - exec_id is used to format log path
    returns:
        - trained model
        - preprocessed validation dataset
    '''
    train = PreprocessedDataset(train_X, train_y, crop_size=config.input_size, resize=config.native_size, horizontal_flip=True, test=False, gpu=config.gpu_id)
    val = PreprocessedDataset(test_X, test_y, crop_size=config.input_size, resize=config.native_size, horizontal_flip=False, test=True, gpu=config.gpu_id)
    train_iter = chainer.iterators.SerialIterator(train, config.batch_size, repeat=True, shuffle=True)
    val_iter = chainer.iterators.SerialIterator(val, config.batch_size, repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=0)
    trainer = training.Trainer(updater, (config.max_epoch, 'epoch'), out=os.path.join(config.log_path, exec_id))
    exts_and_trigs = [
        (
            extensions.Evaluator(val_iter, model, device=config.gpu_id),
            config.val_interval
        ),
        (
            extensions.LogReport(),
            config.log_interval
        ),
        (
            extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'),
            None
        ),
        (
            extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'),
            None
        ),
        (
            extensions.ProgressBar(update_interval=10),
            None
        ),
        (
            extensions.PrintReport([
                'epoch', 'iteration', 'main/loss', 'validation/main/loss',
                'main/accuracy', 'validation/main/accuracy'
            ]),
            config.log_interval
        ),
        (
            extensions.ExponentialShift('lr', 0.5),
            config.lr_decay_interval
        )
    ]
    for ext, trig in exts_and_trigs:
        trainer.extend(ext, trigger=trig)

    trainer.run()
    return model, val


def save_data_info(config, exec_id, train_X, test_X, train_y, test_y):
    df_train = pd.DataFrame({'filename':train_X, 'label':train_y})
    df_val = pd.DataFrame({'filename':test_X, 'label':test_y})
    df_train['for_train'] = True
    df_val['for_train'] = False
    df_train.append(df_val, ignore_index=True).to_csv(os.path.join(config.log_path, exec_id, 'data_info.csv'), index=False)


def save_model(config, model_name, model):
    #save model
    if not os.path.exists(config.trained_model_path):
        os.makedirs(config.trained_model_path)
    serializers.save_hdf5(os.path.join(config.trained_model_path, model_name), model)
    print('model saved')


def fetch_filename_and_labels(config):
    filenames = []
    labels = []
    for idx, c in enumerate(config.category):
        label = idx
        filepath = os.path.join(config.raw_data_path, c)
        for img in os.listdir(filepath):
            filenames.append(os.path.join(filepath, img))
            labels.append(label)
    return np.array(filenames), np.array(labels)


if __name__ == '__main__':
    config = ModelConfig(base_dir=os.path.dirname(__file__))
    execute_train(config)
