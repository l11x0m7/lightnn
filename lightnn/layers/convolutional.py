# -*- encoding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ..base import activations
from ..base.activations import Sigmoid, Relu, Identity
from ..base.initializers import xavier_uniform_initializer
from ..layers.layer import Layer
from ..ops import _check_convolution_layer


class Filter(object):
    def __init__(self, filter_shape, initializer):
        """Filter unit.

        # Params
        filter_shape: (filter_height, filter_width, input_depth).
        initializer: initializer for filters.
        """
        assert len(filter_shape) == 3
        self.filter_shape = filter_shape
        self.__W = initializer(filter_shape)
        self.__b = 0.
        self.__delta_W = np.zeros(filter_shape)
        self.__delta_b = 0.

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def delta_W(self):
        return self.__delta_W

    @property
    def delta_b(self):
        return self.__delta_b

    @W.setter
    def W(self, W):
        self.__W = W

    @b.setter
    def b(self, b):
        self.__b = b

    @delta_W.setter
    def delta_W(self, delta_W):
        self.__delta_W = delta_W

    @delta_b.setter
    def delta_b(self, delta_b):
        self.__delta_b = delta_b

    def update(self):
        self.__W -= self.__delta_W
        self.__b -= self.__delta_b


class Conv2d(Layer):
    """Convolutional Layer.
        For 2d inputs(except depth dimension and batch dimension).
    """
    def __init__(self, filter_size, filter_num, input_shape=None,
                 zero_padding=0, stride=1, activator=Relu, initializer=xavier_uniform_initializer):
        """
        Convolution Layer.

        # Params
        filter_size: (height, width).
        filter_num: the number of filters.
        input_shape: (batch, height, width, depth).
        zero_padding: zero padding number, int or tuple.
        stride: int or tuple, given (stride_height, stride_width) or stride
                        to control the size of output picture.
        activator: activator like tanh or sigmoid or relu.
        initializer: initializer for weights and biases.
        """

        # expand zero padding
        super(Conv2d, self).__init__()
        if isinstance(zero_padding, int):
            zero_padding = (zero_padding, zero_padding)
        # check params
        _check_convolution_layer(filter_size, filter_num, zero_padding, stride)
        self.input_shape = input_shape
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.output_shape = None
        self.zero_padding = zero_padding
        self.activator = activations.get(activator)
        self.initializer = initializer
        self.stride = stride if isinstance(stride, list) or isinstance(stride, tuple) \
                                else (stride, stride)
        if self.input_shape is not None:
            self.connection(None)

    @property
    def delta(self):
        return self.__delta

    @property
    def W(self):
        return [filter.W for filter in self.filters]

    @property
    def b(self):
        return [filter.b for filter in self.filters]

    @property
    def delta_W(self):
        return [filter.delta_W for filter in self.filters]

    @property
    def delta_b(self):
        return [filter.delta_b for filter in self.filters]

    @property
    def params(self):
        return self.W + self.b

    @property
    def grads(self):
        return self.delta_W + self.delta_b

    def call(self, pre_layer=None, *args, **kwargs):
        self.connection(pre_layer)
        return self

    def connection(self, pre_layer):
        if pre_layer is None:
            if self.input_shape is None:
                raise ValueError('input_shape must not be `None` as the first layer.')
        else:
            self.input_shape = pre_layer.output_shape
            self.pre_layer = pre_layer
            pre_layer.next_layer = self

        assert len(self.input_shape) == 4

        # calc output shape
        if self.output_shape is None:
            self.output_shape = self._calc_output_shape(self.input_shape, self.filter_size,
                                        self.stride, self.zero_padding, self.filter_num)
        # add filters
        self.filters = [Filter(list(self.filter_size) + [self.input_shape[3]], self.initializer)
                        for _ in xrange(self.filter_num)]

    def forward(self, input, *args, **kwargs):
        self.input = np.asarray(input)
        assert list(self.input_shape[1:]) == list(self.input.shape[1:])
        self.input_shape = self.input.shape
        self.output_shape[0] = self.input.shape[0]
        # add logit
        self.logit = np.zeros(self.output_shape)
        # add output
        self.output = np.zeros(self.output_shape)
        assert list(self.input.shape[1:]) == list(self.input_shape[1:])
        self.padded_input = self._padding(self.input, self.zero_padding)

        for o_c, filter in enumerate(self.filters):
            filter_W = filter.W
            filter_b = filter.b
            for bn in xrange(self.input_shape[0]):
                self._conv(self.padded_input[bn], filter_W,
                           self.logit[bn,:,:,o_c], filter_b, self.stride)

        self.output = self.activator.forward(self.logit)
        return self.output

    def backward(self, pre_delta_map, *args, **kwargs):
        self.__delta = np.zeros(self.input_shape)
        pre_delta_map = pre_delta_map * self.activator.backward(self.logit)
        expanded_pre_delta_map = self.__expand_sensitive_map(pre_delta_map)
        expanded_batch, expanded_height, expanded_width, expanded_channel = \
                                                            expanded_pre_delta_map.shape
        # expanded_height + 2*pad - filter_height + 1 = input_height
        zero_padding = [0, 0]
        zero_padding[0] = (self.input_shape[1] + self.filter_size[0] - expanded_height - 1) // 2
        zero_padding[1] = (self.input_shape[2] + self.filter_size[1] - expanded_width - 1) // 2
        zero_padding[0] = max(0, zero_padding[0])
        zero_padding[1] = max(0, zero_padding[1])
        padded_delta_map = self._padding(expanded_pre_delta_map, zero_padding)
        for i, filter in enumerate(self.filters):
            rot_weights = np.zeros(filter.W.shape)
            for c in xrange(rot_weights.shape[2]):
                rot_weights[:,:,c] = np.rot90(filter.W[:,:,c], 2)
            delta_a = np.zeros(self.input_shape)
            for i_c in xrange(self.input_shape[3]):
                for bn in xrange(self.input_shape[0]):
                    # calculate delta_{l-1}
                    self._conv(padded_delta_map[bn,:,:,i], rot_weights[:,:,i_c],
                               delta_a[bn,:,:,i_c], 0, (1, 1))
                    # calclulate gradient of w
                    self._conv(self.padded_input[bn,:,:,i_c], expanded_pre_delta_map[bn,:,:,i],
                               filter.delta_W[:,:,i_c], 0, (1, 1))
            filter.delta_W = filter.delta_W / self.input.shape[0]
            filter.delta_b = np.sum(expanded_pre_delta_map[:,:,:,i]) / self.input.shape[0]
            self.__delta += delta_a
        # backward delta_{l-1}
        self.__delta *= self.activator.backward(self.input)
        return self.delta

    def _padding(self, inputs, zero_padding):
        inputs = np.asarray(inputs)
        if list(zero_padding) == [0, 0]:
            return inputs

        if inputs.ndim == 3:
            inputs = inputs[:,:,:,None]

        if inputs.ndim == 4:
            input_batch, input_height, input_width, input_channel = inputs.shape
            padded_input = np.zeros([input_batch, input_height + 2 * zero_padding[0],
                             input_width + 2 * zero_padding[1], input_channel])
            padded_input[:, zero_padding[0]:input_height + zero_padding[0],
                            zero_padding[1]:input_width + zero_padding[1], :] = inputs
            return padded_input
        else:
            raise ValueError('Your input must be a 3-D or 4-D tensor.')

    def _conv(self, inputs, filter_W, outputs, filter_b, stride):
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

        i_height, i_width, _ = inputs.shape
        o_height, o_width = outputs.shape
        stride_height, stride_width = stride
        f_height, f_width, _ = filter_W.shape
        bw = bh = 0
        eh = f_height; ew = f_width
        for idx_height in xrange(o_height):
            for idx_width in xrange(o_width):
                if eh > i_height or ew > i_width:
                    break
                outputs[idx_height,idx_width] = \
                        np.sum(inputs[bh:eh,bw:ew,:] * filter_W) + filter_b
                bw += stride_width
                ew += stride_width
            bh += stride_height
            eh += stride_height
            bw = 0; ew = f_width

    def __expand_sensitive_map(self, pre_delta_map):
        batch, height, width, depth = pre_delta_map.shape
        stride_height, stride_width = self.stride
        expanded_height = (height - 1) * stride_height + 1
        expanded_width = (width - 1) * stride_width + 1

        expanded_pre_delta_map = np.zeros([batch, expanded_height, expanded_width, depth])

        for i in xrange(height):
            for j in xrange(width):
                expanded_pre_delta_map[:, stride_height * i,
                            stride_width * j, :] = pre_delta_map[:,i,j,:]
        return expanded_pre_delta_map

    def _calc_output_size(self, input_len, filter_len, stride, zero_padding):
        return (input_len + 2 * zero_padding - filter_len) // stride + 1

    def _calc_output_shape(self, input_shape, filter_shape, stride, zero_padding, filter_num):
        output_height = self._calc_output_size(input_shape[1], filter_shape[0],
                                                    stride[0], zero_padding[0])
        output_width = self._calc_output_size(input_shape[2], filter_shape[1],
                                                   stride[1], zero_padding[1])
        output_channel = filter_num
        return [input_shape[0], output_height, output_width, output_channel]
