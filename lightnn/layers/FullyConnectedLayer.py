# -*- encoding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
from ..base.BasicFunctions import Sigmoid
from ..base.Initializers import xavier_weight_initializer


class FullyConnectedLayer(object):
    def __init__(self, input_size, output_size, activator=Sigmoid, initializer=xavier_weight_initializer):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        self.W = initializer([self.output_size, self.input_size])
        self.b = initializer([self.output_size])

    def forward(self, inputs):
        """
        :param inputs: 2-D tensors, row represents samples, col represents features
        :return: None
        """
        self.input = inputs
        self.logit = np.dot(self.W, np.asarray(inputs).T).T + self.b
        self.output = self.activator.forward(self.logit)
        return self.output

    def backward(self, pre_delta, pre_W):
        self.delta = np.dot(pre_W.T, pre_delta) * self.activator.backward(self.logit)
        self.delta_W = np.dot(self.delta.reshape([-1, 1]), self.input.reshape([1, -1]))
        self.delta_b = self.delta
        return self.delta

    def get_W(self):
        return self.W

    def get_b(self):
        return self.b

    def update_delta(self, delta):
        self.delta = delta
        self.delta_W = np.dot(self.delta.reshape([-1, 1]), self.input.reshape([1, -1]))
        self.delta_b = self.delta

    def step(self, delta_W=None, delta_b=None):
        if not hasattr(self, 'delta_W') or not hasattr(self, 'delta_b'):
            raise AttributeError('You must execute backward first.')
        delta_W = self.delta_W if delta_W is None else delta_W
        delta_b = self.delta_b if delta_b is None else delta_b
        self.W -= delta_W
        self.b -= delta_b






