# -*- encoding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
from ..base.BasicFunctions import sigmoid, delta_sigmoid, identity, delta_identity
from ..base.BasicFunctions import Sigmoid, Relu, Identity
from ..base.Costs import CECost
from ..base.Initializers import xavier_weight_initializer


class MaxPoolingLayer(object):
    def __init__(self, input_width, input_height, input_channel, window_width, window_height, stride, zero_padding):
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.window_width = window_width
        self.window_height = window_height
        self.zero_padding = zero_padding

        self.input = np.zeros([input_width, input_height, input_channel])

        self.stride = stride
        self.output_width = (self.input_width + self.zero_padding * 2 - self.window_width) // self.stride[0] + 1
        self.output_height = (self.input_height + self.zero_padding * 2 - self.window_height) // self.stride[1] + 1
        self.output = np.zeros([self.output_width, self.output_height, self.input_channel])

        self.delta = np.zeros([input_width, input_height, input_channel])


    def padding(self, inputs, zero_padding):
        inputs = np.asarray(inputs)
        if zero_padding == 0:
            return inputs

        if inputs.ndim == 2:
            inputs = inputs[:,:,None]

        if inputs.ndim == 3:
            input_width, input_height, input_channel = inputs.shape
            padded_input = np.zeros([input_width + 2 * zero_padding,
                             input_height + 2 * zero_padding, input_channel])
            padded_input[zero_padding:input_width + zero_padding,
                            zero_padding:input_height + zero_padding, :] = inputs
            return padded_input
        else:
            raise ValueError('Your input must be a 2-D or 3-D tensor.')

    def forward(self, inputs):
        inputs = np.asarray(inputs)
        if inputs.ndim == 2:
            inputs = inputs[:,:,None]
        if inputs.ndim == 3:
            self.input = inputs
        else:
            raise ValueError('Your input must be a 2-D or 3-D tensor.')

        self.padded_input = self.padding(self.input, self.zero_padding)

        self.max_ind = np.zeros(list(self.output.shape) + [2], dtype=int)
        for idx_c in xrange(self.input_channel):
            wb = hb = 0
            we = self.window_width
            he = self.window_height
            for i in xrange(self.output_width):
                for j in xrange(self.output_height):
                    self.output[i,j,idx_c] = np.max(self.padded_input[wb:we,hb:he,idx_c])
                    max_ind = np.argmax(self.padded_input[wb:we,hb:he,idx_c])
                    max_x, max_y = max_ind / self.window_width, max_ind % self.window_width
                    self.max_ind[i,j,idx_c] = [max_x + wb, max_y + hb]
                    hb += self.stride[1]
                    he += self.stride[1]
                wb += self.stride[0]
                we += self.stride[0]
                hb = 0; he = self.window_height
        return self.output

    def backward(self, pre_delta_map):
        for idx_c in xrange(self.input_channel):
            for i in xrange(self.output_width):
                for j in xrange(self.output_height):
                    x, y = self.max_ind[i,j,idx_c]
                    if x < self.zero_padding or x >= self.input_width + self.zero_padding:
                        continue
                    if y < self.zero_padding or y >= self.input_height + self.zero_padding:
                        continue
                    x -= self.zero_padding
                    y -= self.zero_padding
                    self.delta[x,y,idx_c] += pre_delta_map[i,j,idx_c]
        return self.delta

class AvgPoolingLayer(object):
    def __init__(self, input_width, input_height, input_channel, window_width, window_height, stride, zero_padding):
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.window_width = window_width
        self.window_height = window_height
        self.zero_padding = zero_padding

        self.input = np.zeros([input_width, input_height, input_channel])

        self.stride = stride
        self.output_width = (self.input_width + self.zero_padding * 2 - self.window_width) // self.stride[0] + 1
        self.output_height = (self.input_height + self.zero_padding * 2 - self.window_height) // self.stride[1] + 1
        self.output = np.zeros([self.output_width, self.output_height, self.input_channel])

        self.delta = np.zeros([input_width, input_height, input_channel])


    def padding(self, inputs, zero_padding):
        inputs = np.asarray(inputs)
        if zero_padding == 0:
            return inputs

        if inputs.ndim == 2:
            inputs = inputs[:,:,None]

        if inputs.ndim == 3:
            input_width, input_height, input_channel = inputs.shape
            padded_input = np.zeros([input_width + 2 * zero_padding,
                             input_height + 2 * zero_padding, input_channel])
            padded_input[zero_padding:input_width + zero_padding,
                            zero_padding:input_height + zero_padding, :] = inputs
            return padded_input
        else:
            raise ValueError('Your input must be a 2-D or 3-D tensor.')

    def forward(self, inputs):
        inputs = np.asarray(inputs)
        if inputs.ndim == 2:
            inputs = inputs[:,:,None]
        if inputs.ndim == 3:
            self.input = inputs
        else:
            raise ValueError('Your input must be a 2-D or 3-D tensor.')

        self.padded_input = self.padding(self.input, self.zero_padding)

        for idx_c in xrange(self.input_channel):
            wb = hb = 0
            we = self.window_width
            he = self.window_height
            for i in xrange(self.output_width):
                for j in xrange(self.output_height):
                    self.output[i,j,idx_c] = np.sum(self.padded_input[wb:we,hb:he,idx_c])\
                                                / float(self.window_height * self.window_width)
                    hb += self.stride[1]
                    he += self.stride[1]
                wb += self.stride[0]
                we += self.stride[0]
                hb = 0; he = self.window_height
        return self.output

    def backward(self, pre_delta_map):
        for idx_c in xrange(self.input_channel):
            wb = hb = 0
            we = self.window_width
            he = self.window_height
            for i in xrange(self.output_width):
                for j in xrange(self.output_height):
                    self.delta[wb:we,hb:he,idx_c] += (pre_delta_map[i,j,idx_c] \
                            / float(self.window_height * self.window_width))
                    hb += self.stride[1]
                    he += self.stride[1]
                wb += self.stride[0]
                we += self.stride[0]
                hb = 0; he = self.window_height
        return self.delta


