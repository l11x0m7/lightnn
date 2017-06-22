# -*- encoding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class RecurrentLayer(object):
    def __init__(self, input_size, hidden_size, activator, initializer):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activator = activator

        self.state = [np.zeros([self.hidden_size, 1])]
        self.U = initializer([ self.hidden_size, self.input_size])
        self.W = initializer([self.hidden_size, self.hidden_size])
        self.b = np.zeros([self.hidden_size, 1])
        self.times = 0

    def forward(self, input):
        self.input = input
        self.times += 1
        self.output = self.activator(
                np.dot(self.U, self.input) + np.dot(self.W, self.state[-1]) + self.b)
        self.state.append(self.output)
        return self.output

    def backward(self, pre_delta):
        pass

    def calc_delta(self, pre_delta):
        pass
