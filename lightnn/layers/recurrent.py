# -*- encoding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .layer import Layer
from ..base.initializers import glorot_uniform_initializer, orthogonal_initializer
from ..base import activations


class Recurrent(Layer):
    """Abstract base class for recurrent layers.

    Do not use in a model -- it's not a valid layer!

    Use its children classes `LSTM`, `GRU` and `SimpleRNN` instead.
    All recurrent layers (`LSTM`, `GRU`, `SimpleRNN`) also
    follow the specifications of this class and accept
    the keyword arguments listed below.
    """

    def __init__(self, output_dim, input_shape=None,
                 activator='tanh',  kernel_initializer=glorot_uniform_initializer,
                 recurrent_initializer=orthogonal_initializer,
                 return_sequences=False, **kwargs):
        # input_shape:(batch size, sequence length, input dimension)
        super(Recurrent, self).__init__()
        self.output_dim = output_dim
        self.activator = activations.get(activator)
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.return_sequences = return_sequences
        self.input_dim = input_shape[-1]
        self.input_shape = input_shape
        self.output_shape = None
        self.connection(None)

    def connection(self, pre_layer):
        if pre_layer is None:
            if self.input_shape is None:
                raise ValueError('input_shape must not be `None` as the first layer.')
        else:
            self.input_shape = pre_layer.output_shape
            self.pre_layer = pre_layer
            pre_layer.next_layer = self

        if self.return_sequences:
            self.output_shape = [self.input_shape[0], self.input_shape[1], self.output_dim]
        else:
            self.output_shape = [self.input_shape[0], self.output_dim]


class SimpleRNN(Recurrent):
    def __init__(self, hidden_size, input_shape=None, use_bias=True, **kwargs):
        super(SimpleRNN, self).__init__(hidden_size, input_shape, **kwargs)
        self.use_bias = use_bias
        self.W = None
        self.U = None
        self.b = None
        self.delta_W = None
        self.delta_U = None
        self.delta_b = None
        self.states = list()
        self.connection(None)

    @property
    def W(self):
        return self.__W

    @W.setter
    def W(self, W):
        self.__W = W

    @property
    def U(self):
        return self.__U

    @U.setter
    def U(self, U):
        self.__U = U

    @property
    def b(self):
        return self.__b

    @b.setter
    def b(self, b):
        self.__b = b

    @property
    def delta_W(self):
        return self.__delta_W

    @delta_W.setter
    def delta_W(self, delta_W):
        self.__delta_W = delta_W

    @property
    def delta_U(self):
        return self.__delta_U

    @delta_U.setter
    def delta_U(self, delta_U):
        self.__delta_U = delta_U

    @property
    def delta_b(self):
        return self.__delta_b

    @delta_b.setter
    def delta_b(self, delta_b):
        self.__delta_b = delta_b

    def reset(self):
        self.states = list()

    def connection(self, pre_layer):
        super(SimpleRNN, self).connection(pre_layer)
        self.W = self.kernel_initializer((self.input_dim, self.output_dim))
        self.U = self.recurrent_initializer((self.output_dim, self.output_dim))
        self.b = np.zeros((self.output_dim,))

    def forward(self, inputs, *args, **kwargs):
        # inputs: batch_size, time_step, out_dim
        inputs = np.asarray(inputs)
        self.input = inputs
        assert inputs.shape[1:] == self.input_shape[1:]
        self.input_shape = inputs.shape
        self.output_shape[0] = self.input_shape[0]
        nb_batch, nb_seq, nb_input_dim = self.input_shape
        self.outputs = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.logits = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.states.append(np.zeros((nb_batch, self.output_dim)))
        for t in xrange(nb_seq):
            self.outputs[:,t,:] = self.input[:,t,:].dot(self.W) + self.states[-1].dot(self.U)
            if self.use_bias:
                self.outputs[:,t,:] += self.b
            self.logits[:,t,:] = self.outputs[:,t,:]
            self.outputs[:,t,:] = self.activator.forward(self.outputs[:,t,:])
            self.states.append(self.outputs[:,t,:])

        if self.return_sequences:
            return self.outputs
        else:
            return self.outputs[:,-1,:]

    def backward(self, pre_delta, *args, **kwargs):
        self.delta_W = np.zeros(self.W.shape)
        self.delta_U = np.zeros(self.U.shape)
        if self.use_bias:
            self.delta_b = np.zeros(self.b.shape)
        nb_batch, nb_seq, nb_input_dim = self.input_shape
        self.delta = np.zeros(self.input_shape)
        if self.return_sequences:
            # TODO
            pass
        else:
            # 同一层的误差传递（从T到1）
            time_delta = pre_delta * self.activator.backward(self.logits[:,-1,:])
            for t in xrange(nb_seq - 1, -1, -1):
                # 求U的梯度
                self.delta_U += np.dot(self.states[t].T, time_delta)
                # 求W的梯度
                self.delta_W += np.dot(self.input[:,t,:].T, time_delta)
                # 求b的梯度
                if self.use_bias:
                    self.delta_b += np.mean(time_delta, axis=0)
                # 求传到上一层的误差,timewise
                self.delta[:,t,:] = np.dot(time_delta, self.W.T)
                # 求同一层的误差,layerwise
                if t > 0:
                    for bn in xrange(nb_batch):
                        time_delta[bn,:] = np.dot(
                                time_delta[bn,:], np.dot(
                                np.diag(self.activator.backward(self.logits[bn,t-1,:])), self.U).T)
        return self.delta
