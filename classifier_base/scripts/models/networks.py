#!/usr/bin/env python
# -- coding:utf-8 --

import chainer
import chainer.functions as F
import chainer.links as L

class FineTuneResNet(chainer.Chain):
    def __init__(self, path, layer, class_num):
        super(FineTuneResNet, self).__init__()
        with self.init_scope():
            self.base_model = path.split('/')[-1]
            self.resnet = chainer.links.model.vision.resnet.ResNetLayers(path, layer)
            initializer = chainer.initializers.HeNormal()
            self.linear = L.Linear(2048, class_num, initialW=initializer)

    def __call__(self, x, t):
        feature = self.resnet(x, layers=['pool5'])['pool5']
        h = F.dropout(feature, ratio=.5)
        h = self.linear(h)
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

    def infer(self, x):
        feature = self.resnet(x, layers=['pool5'])['pool5']
        h = self.linear(feature)
        return F.softmax(h, axis=1)
