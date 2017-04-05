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


class Filter(object):
    def __init__(self, filter_width, filter_height, filter_channel, initializer):
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_channel = filter_channel
        self.W = initializer([filter_width, filter_height, filter_channel])
        self.b = 0.
        self.delta_W = np.zeros([filter_width, filter_height, filter_channel])
        self.delta_b = 0.

    def get_W(self):
        return self.W

    def get_b(self):
        return self.b

    def update(self):
        self.W -= self.delta_W
        self.b -= self.delta_b


class ConvLayer(object):
    def __init__(self, input_width, input_height, input_channel,
                 filter_width, filter_height, filter_num,
                 zero_padding, stride, activator, initializer, lr=1e-1):
        """
        Convolution Layer
        :param input_width: the input picture's width
        :param input_height: the input picture's height
        :param input_channel: the input pictures's channel number
        :param filter_width: the filter's width
        :param filter_height: the filter's height
        :param filter_num: the number of filters used in this layer
        :param zero_padding: zero padding number
        :param stride: given the [stride_width, stride_height] to control the size of output picture
        :param activator: activator like tanh or sigmoid or relu
        """

        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_num = filter_num
        self.filters = [Filter(filter_width, filter_height, input_channel, initializer)
                            for _ in xrange(filter_num)]
        self.output_width = self.calc_output_size(input_width, filter_width, stride[0], zero_padding)
        self.output_height = self.calc_output_size(input_height, filter_height, stride[1], zero_padding)
        self.output_channel = filter_num
        self.output = np.zeros([self.output_width, self.output_height, self.output_channel])
        self.zero_padding = zero_padding
        self.activator = activator
        self.stride = stride
        self.lr = lr

    def calc_output_size(self, input_len, filter_len, stride, zero_padding):
        return (input_len + 2 * zero_padding - filter_len) // stride + 1

    def forward(self, input):
        self.input = input
        self.padded_input = self.padding(self.input, self.zero_padding)

        for o_c, filter in enumerate(self.filters):
            filter_W = filter.get_W()
            filter_b = filter.get_b()
            self.conv(self.padded_input, filter_W, self.output[:,:,o_c], filter_b, self.stride)

        self.output = self.activator.forward(self.output)
        return self.output

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

    def conv(self, inputs, filter_W, outputs, filter_b, stride):
        inputs = np.asarray(inputs)
        if inputs.ndim == 2:
            inputs = inputs[:,:,None]
        elif inputs.ndim == 3:
            inputs = inputs
        else:
            raise ValueError('Your input must be a 2-D or 3-D tensor.')
        if filter_W.ndim == 2:
            filter_W = filter_W[:,:,None]
        elif filter_W.ndim == 3:
            filter_W = filter_W
        else:
            raise ValueError('Your filter_W must be a 2-D or 3-D tensor.')


        o_width, o_height, _ = self.output.shape
        stride_width, stride_height = stride
        f_width, f_height, _ = filter_W.shape
        bw = bh = 0
        ew = f_width; eh = f_height
        for idx_width in xrange(o_width):
            for idx_height in xrange(o_height):
                outputs[idx_width,idx_height] = \
                        np.sum(inputs[bw:ew,bh:eh,:] * filter_W) + filter_b
                bh += stride_height
                eh += stride_height
            bw += stride_width
            ew += stride_width
            bh = 0; eh = f_height


    def backward(self, pre_delta_map, activator):
        expanded_pre_delta_map = self.__expand_sensitive_map(pre_delta_map)
        expanded_width, expanded_height, expanded_channel = expanded_pre_delta_map.shape
        # expanded_width + 2*pad - filter_width + 1 = input_width
        zero_padding = (self.input_width + self.filter_width - expanded_width - 1) // 2
        padded_delta_map = self.padding(expanded_pre_delta_map, zero_padding)

        self.delta = np.zeros((self.input_height, self.input_width,
                               self.input_channel))

        for i, filter in enumerate(self.filters):
            rot_weights = filter.get_W()
            for c in xrange(rot_weights.shape[2]):
                rot_weights[:,:,c] = np.rot90(rot_weights[:,:,c], 2)

            delta = np.zeros((self.input_height, self.input_width,
                               self.input_channel))
            for i_c in xrange(self.input_channel):
                self.conv(padded_delta_map[:,:,i], rot_weights[:,:,i_c], delta[:,:,i_c], 0, [1, 1])
                self.conv(self.padded_input[:,:,i_c], expanded_pre_delta_map[:,:,i], filter.delta_W[:,:,i_c], 0, [1, 1])
            filter.delta_b = np.sum(expanded_pre_delta_map[:,:,i])

            self.delta += delta

        self.delta *= activator.backward(self.input)

        return self.delta

    def __expand_sensitive_map(self, pre_delta_map):
        width, height, depth = pre_delta_map.shape
        stride_width, stride_height = self.stride
        expanded_width = (width - 1) * stride_width + 1
        expanded_height = (height - 1) * stride_height + 1

        expanded_pre_delta_map = np.zeros([expanded_width, expanded_height, depth])

        for i in xrange(width):
            for j in xrange(height):
                expanded_pre_delta_map[stride_width * i,
                            stride_height * j, :] = pre_delta_map[i,j,:]
        return expanded_pre_delta_map


    def update(self):
        for filter in self.filters:
            filter.delta_W *= self.lr
            filter.delta_b *= self.lr
            filter.update()






