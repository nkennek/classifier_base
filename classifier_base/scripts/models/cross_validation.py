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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

import matplotlib as mpl
#for PlotReport on server
mpl.use('Agg')

from classifier_base.scripts.models.data_model import PreprocessedDataset
from classifier_base.scripts.models.model_config import ModelConfig
from classifier_base.scripts.models.predict import get_newest_modelpath
from classifier_base.scripts.models.train_models import train, fetch_filename_and_labels, save_data_info


def cross_validation(config, n_iter=10):
    timestamp = dt.now().strftime('%Y-%m-%d_%H:%M')
    exec_id = f"CrossValidation_{timestamp}"
    filenames, labels = fetch_filename_and_labels(config)

    # Stratified K-Folds cross-validator
    skf = StratifiedKFold(n_splits=n_iter, random_state=config.random_state, shuffle=False)
    fold = 1 # initial value
    records = []
    accs    = []
    idxes   = {}
    for train_idx, test_idx in skf.split(filenames, labels):
        print('{}:{} time iteration:'.format(dt.now(), fold))
        model, optimizer = config.setup()
        train_X, test_X = filenames[train_idx], filenames[test_idx]
        train_y, test_y = labels[train_idx], labels[test_idx]
        acc, record = run_1fold(config, model, optimizer, train_X, test_X, train_y, test_y, exec_id, fold)
        save_data_info(config, exec_id+'/'+str(fold), train_X, test_X, train_y, test_y)
        accs.append(acc)
        records.append(record)
        fold += 1

    pd.concat(records, axis=1).to_csv(os.path.join(config.log_path, exec_id, 'raw_record.csv'), index=False)
    pd.Series(accs).to_csv(os.path.join(config.log_path, exec_id, 'accuracies.csv'))
    config.dump(exec_id)


def run_1fold(config, model, optimizer, train_X, test_X, train_y, test_y, exec_id, nth):
    model, val = train(config, model, optimizer, train_X, test_X, train_y, test_y, exec_id+'/'+str(nth))
    serializers.save_hdf5(os.path.join(config.log_path, exec_id+'/'+str(nth), 'model.h5'), model)
    # validation result
    with chainer.using_config('train', False):
        model.to_cpu()
        test_set = val[np.arange(len(test_X))]
        imgs, labels = np.array([t[0] for t in test_set]), np.array([t[1] for t in test_set])
        raw_pred = model.infer(imgs).data
        pred_labels = np.argmax(raw_pred, axis=1)
        print(classification_report(labels, pred_labels, target_names=config.category))
        return accuracy_score(labels, pred_labels), pd.DataFrame({'fold':nth, 'filename': test_X, 'label': labels, 'pred_0': raw_pred[:, 0], 'pred_1': raw_pred[:, 1]})


if __name__ == '__main__':
    config = ModelConfig(base_dir=os.path.dirname(__file__))
    cross_validation(config)
