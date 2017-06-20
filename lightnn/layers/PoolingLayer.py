# -*- encoding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class MaxPoolingLayer(object):
    def __init__(self, input_height, input_width, input_channel, window_height, window_width, stride=1, zero_padding=0):
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.window_height = window_height
        self.window_width = window_width
        if isinstance(zero_padding, int):
            zero_padding = (zero_padding, zero_padding)
        self.zero_padding = zero_padding

        self.input = np.zeros([input_height, input_width, input_channel])

        self.stride = stride
        self.output_height = (self.input_height + self.zero_padding[0] * 2 - self.window_height) // self.stride[0] + 1
        self.output_width = (self.input_width + self.zero_padding[1] * 2 - self.window_width) // self.stride[1] + 1
        self.output = np.zeros([self.output_height, self.output_width, self.input_channel])

        self.__delta = np.zeros([input_height, input_width, input_channel])

    @property
    def delta(self):
        return self.__delta

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
            we = self.window_height
            he = self.window_width
            for i in xrange(self.output_height):
                for j in xrange(self.output_width):
                    self.output[i,j,idx_c] = np.max(self.padded_input[wb:we,hb:he,idx_c])
                    max_ind = np.argmax(self.padded_input[wb:we,hb:he,idx_c])
                    max_x, max_y = max_ind / self.window_height, max_ind % self.window_height
                    self.max_ind[i,j,idx_c] = [max_x + wb, max_y + hb]
                    hb += self.stride[1]
                    he += self.stride[1]
                wb += self.stride[0]
                we += self.stride[0]
                hb = 0; he = self.window_width
        return self.output

    def backward(self, pre_delta_map):
        for idx_c in xrange(self.input_channel):
            for i in xrange(self.output_height):
                for j in xrange(self.output_width):
                    x, y = self.max_ind[i,j,idx_c]
                    if x < self.zero_padding[0] or x >= self.input_height + self.zero_padding[0]:
                        continue
                    if y < self.zero_padding[1] or y >= self.input_width + self.zero_padding[1]:
                        continue
                    x -= self.zero_padding[0]
                    y -= self.zero_padding[1]
                    self.__delta[x,y,idx_c] += pre_delta_map[i,j,idx_c]
        return self.__delta

    def padding(self, inputs, zero_padding):
        inputs = np.asarray(inputs)
        if list(zero_padding) == [0, 0]:
            return inputs

        if inputs.ndim == 2:
            inputs = inputs[:,:,None]

        if inputs.ndim == 3:
            input_height, input_width, input_channel = inputs.shape
            padded_input = np.zeros([input_height + 2 * zero_padding[0],
                                     input_width + 2 * zero_padding[1], input_channel])
            padded_input[zero_padding[0]:input_height + zero_padding[0],
            zero_padding[1]:input_width + zero_padding[1], :] = inputs
            return padded_input
        else:
            raise ValueError('Your input must be a 2-D or 3-D tensor.')


class AvgPoolingLayer(object):
    def __init__(self, input_height, input_width, input_channel, window_height, window_width, stride=1, zero_padding=0):
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.window_height = window_height
        self.window_width = window_width
        if isinstance(zero_padding, int):
            zero_padding = [zero_padding, zero_padding]
        self.zero_padding = zero_padding

        self.input = np.zeros([input_height, input_width, input_channel])

        self.stride = stride
        self.output_height = (self.input_height + self.zero_padding[0] * 2 - self.window_height) // self.stride[0] + 1
        self.output_width = (self.input_width + self.zero_padding[1] * 2 - self.window_width) // self.stride[1] + 1
        self.output = np.zeros([self.output_height, self.output_width, self.input_channel])

        self.__delta = np.zeros([input_height, input_width, input_channel])

    @property
    def delta(self):
        return self.__delta

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
            we = self.window_height
            he = self.window_width
            for i in xrange(self.output_height):
                for j in xrange(self.output_width):
                    self.output[i,j,idx_c] = np.sum(self.padded_input[wb:we,hb:he,idx_c])\
                                                / float(self.window_width * self.window_height)
                    hb += self.stride[1]
                    he += self.stride[1]
                wb += self.stride[0]
                we += self.stride[0]
                hb = 0; he = self.window_width
        return self.output

    def backward(self, pre_delta_map):
        for idx_c in xrange(self.input_channel):
            wb = hb = 0
            we = self.window_height
            he = self.window_width
            for i in xrange(self.output_height):
                for j in xrange(self.output_width):
                    self.__delta[wb:we,hb:he,idx_c] += (pre_delta_map[i,j,idx_c] \
                            / float(self.window_width * self.window_height))
                    hb += self.stride[1]
                    he += self.stride[1]
                wb += self.stride[0]
                we += self.stride[0]
                hb = 0; he = self.window_width
        return self.__delta

    def padding(self, inputs, zero_padding):
        inputs = np.asarray(inputs)
        if list(zero_padding) == [0, 0]:
            return inputs

        if inputs.ndim == 2:
            inputs = inputs[:,:,None]

        if inputs.ndim == 3:
            input_height, input_width, input_channel = inputs.shape
            padded_input = np.zeros([input_height + 2 * zero_padding[0],
                                     input_width + 2 * zero_padding[1], input_channel])
            padded_input[zero_padding[0]:input_height + zero_padding[0],
            zero_padding[1]:input_width + zero_padding[1], :] = inputs
            return padded_input
        else:
            raise ValueError('Your input must be a 2-D or 3-D tensor.')
