# -*- encoding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .layer import Layer
from ..base import initializers
from ..base import activations


class Recurrent(Layer):
    """Abstract base class for recurrent layers.

    Do not use in a model -- it's not a valid layer!

    Use its children classes `LSTM`, `GRU` and `SimpleRNN` instead.
    All recurrent layers (`LSTM`, `GRU`, `SimpleRNN`) also
    follow the specifications of this class and accept
    the keyword arguments listed below.
    """

    def __init__(self, output_dim,
                 input_shape=None,
                 return_sequences=False,
                 **kwargs):
        # input_shape:(batch size, sequence length, input dimension)
        super(Recurrent, self).__init__()
        self.output_dim = output_dim
        self.return_sequences = return_sequences
        self.input_shape = input_shape
        self.output_shape = None
        if input_shape is not None:
            self.input_dim = input_shape[-1]

    def connection(self, pre_layer):
        if pre_layer is None:
            if self.input_shape is None:
                raise ValueError('input_shape must not be `None` as the first layer.')
        else:
            self.input_shape = pre_layer.output_shape
            self.pre_layer = pre_layer
            pre_layer.next_layer = self

        self.input_dim = self.input_shape[-1]

        if self.return_sequences:
            self.output_shape = [self.input_shape[0], self.input_shape[1], self.output_dim]
        else:
            self.output_shape = [self.input_shape[0], self.output_dim]


class SimpleRNN(Recurrent):
    """Simple RNN unit.

        Fully-connected RNN where the output is to be fed back to input.
    """

    def __init__(self, output_dim,
                 input_shape=None,
                 activator='tanh',
                 kernel_initializer='glorot_uniform_initializer',
                 recurrent_initializer='orthogonal_initializer',
                 use_bias=True,
                 return_sequences=False,
                 **kwargs):
        super(SimpleRNN, self).__init__(output_dim, input_shape, return_sequences, **kwargs)
        self.activator = activations.get(activator)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.use_bias = use_bias
        self.W = None
        self.U = None
        self.b = None
        self.delta_W = None
        self.delta_U = None
        self.delta_b = None
        self.states = list()
        if input_shape is not None:
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

    @property
    def params(self):
        if self.use_bias:
            return [self.W, self.U, self.b]
        return [self.W, self.U]

    @property
    def grads(self):
        if self.use_bias:
            return [self.delta_W, self.delta_U, self.delta_b]
        return [self.delta_W, self.delta_U]

    def reset(self):
        self.states = list()

    def call(self, pre_layer=None, *args, **kwargs):
        self.connection(pre_layer)
        return self

    def connection(self, pre_layer):
        super(SimpleRNN, self).connection(pre_layer)
        self.W = self.kernel_initializer((self.input_dim, self.output_dim))
        self.U = self.recurrent_initializer((self.output_dim, self.output_dim))
        self.b = np.zeros((self.output_dim,))

    def forward(self, inputs, *args, **kwargs):
        # clear states
        self.reset()
        # inputs: batch_size, time_step, out_dim
        inputs = np.asarray(inputs)
        self.inputs = inputs
        assert list(inputs.shape[1:]) == list(self.input_shape[1:])
        self.input_shape = inputs.shape
        self.output_shape[0] = self.input_shape[0]
        nb_batch, nb_seq, nb_input_dim = self.input_shape
        self.outputs = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.logits = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.states.append(np.zeros((nb_batch, self.output_dim)))
        for t in xrange(nb_seq):
            self.outputs[:,t,:] = self.inputs[:,t,:].dot(self.W) + self.states[-1].dot(self.U)
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
            assert len(pre_delta.shape) == 3
            # 同一层的误差传递（从T到1）,此处的time_delta为delta_E/delta_z
            time_delta = pre_delta[:,nb_seq-1,:] * self.activator.backward(self.logits[:,nb_seq-1,:])
        else:
            assert len(pre_delta.shape) == 2
            # 同一层的误差传递（从T到1）,此处的time_delta为delta_E/delta_z
            time_delta = pre_delta * self.activator.backward(self.logits[:,-1,:])
        for t in xrange(nb_seq - 1, -1, -1):
            # 求U的梯度
            self.delta_U += np.dot(self.states[t].T, time_delta) / nb_batch
            # 求W的梯度
            self.delta_W += np.dot(self.inputs[:,t,:].T, time_delta) / nb_batch
            # 求b的梯度
            if self.use_bias:
                self.delta_b += np.mean(time_delta, axis=0)
            # 求传到上一层的误差,layerwise
            self.delta[:,t,:] = np.dot(time_delta, self.W.T)
            # 求同一层不同时间的误差,timewise
            if t > 0:
                # 下面两种计算同层不同时间误差的方法等效
                # 方法1
                time_delta = np.asarray(
                    map(
                    np.dot, *(time_delta, np.asarray(map(
                    lambda logit:(self.activator.backward(logit) * self.U.T)
                    , self.logits[:,t-1,:])))
                    )
                )
                # 方法2
                # for bn in xrange(nb_batch):
                #     time_delta[bn,:] = np.dot(
                #     time_delta[bn,:], np.dot(
                #     np.diag(self.activator.backward(self.logits[bn,t-1,:])),
                #             self.U).T)
                if self.return_sequences:
                    time_delta += pre_delta[:,t - 1,:] * \
                             self.activator.backward(self.logits[:,t - 1,:])
        self.reset()
        return self.delta


class LSTM(Recurrent):
    """Long-Short Term Memory unit - Hochreiter 1997.

       For a step-by-step description of the algorithm, see
       [this tutorial](http://deeplearning.net/tutorial/lstm.html).

       References:
       1. LSTM: A Search Space Odyssey
          (https://arxiv.org/pdf/1503.04069.pdf)
       2. Backpropogating an LSTM: A Numerical Example
          (https://blog.aidangomez.ca/2016/04/17/
                    Backpropogating-an-LSTM-A-Numerical-Example/)
       3. LSTM(https://github.com/nicodjimenez/lstm)
    """

    def __init__(self, output_dim,
                 input_shape=None,
                 activator='tanh',
                 recurrent_activator='sigmoid',
                 kernel_initializer='glorot_uniform_initializer',
                 recurrent_initializer='orthogonal_initializer',
                 use_bias=True,
                 return_sequences=False,
                 **kwargs):
        super(LSTM, self).__init__(output_dim, input_shape, return_sequences, **kwargs)
        self.use_bias = use_bias
        self.activator = activations.get(activator)
        self.recurrent_activator = activations.get(recurrent_activator)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)

        # input gate
        self.__W_i = None
        self.__U_i = None
        self.__b_i = None
        self.__delta_W_i = None
        self.__delta_U_i = None
        self.__delta_b_i = None

        # forget gate
        self.__W_f = None
        self.__U_f = None
        self.__b_f = None
        self.__delta_W_f = None
        self.__delta_U_f = None
        self.__delta_b_f = None

        # output gate
        self.__W_o = None
        self.__U_o = None
        self.__b_o = None
        self.__delta_W_o = None
        self.__delta_U_o = None
        self.__delta_b_o = None

        # cell memory(long term)
        self.__W_c = None
        self.__U_c = None
        self.__b_c = None
        self.__delta_W_c = None
        self.__delta_U_c = None
        self.__delta_b_c = None

        if input_shape is not None:
            self.connection(None)

    @property
    def W_i(self):
        return self.__W_i

    @W_i.setter
    def W_i(self, W_i):
        self.__W_i = W_i

    @property
    def U_i(self):
        return self.__U_i

    @U_i.setter
    def U_i(self, U_i):
        self.__U_i = U_i

    @property
    def b_i(self):
        return self.__b_i

    @b_i.setter
    def b_i(self, b_i):
        self.__b_i = b_i

    @property
    def delta_W_i(self):
        return self.__delta_W_i

    @delta_W_i.setter
    def delta_W_i(self, delta_W_i):
        self.__delta_W_i = delta_W_i

    @property
    def delta_U_i(self):
        return self.__delta_U_i

    @delta_U_i.setter
    def delta_U_i(self, delta_U_i):
        self.__delta_U_i = delta_U_i

    @property
    def delta_b_i(self):
        return self.__delta_b_i

    @delta_b_i.setter
    def delta_b_i(self, delta_b_i):
        self.__delta_b_i = delta_b_i

    @property
    def W_o(self):
        return self.__W_o

    @W_o.setter
    def W_o(self, W_o):
        self.__W_o = W_o

    @property
    def U_o(self):
        return self.__U_o

    @U_o.setter
    def U_o(self, U_o):
        self.__U_o = U_o

    @property
    def b_o(self):
        return self.__b_o

    @b_o.setter
    def b_o(self, b_o):
        self.__b_o = b_o

    @property
    def delta_W_o(self):
        return self.__delta_W_o

    @delta_W_o.setter
    def delta_W_o(self, delta_W_o):
        self.__delta_W_o = delta_W_o

    @property
    def delta_U_o(self):
        return self.__delta_U_o

    @delta_U_o.setter
    def delta_U_o(self, delta_U_o):
        self.__delta_U_o = delta_U_o

    @property
    def delta_b_o(self):
        return self.__delta_b_o

    @delta_b_o.setter
    def delta_b_o(self, delta_b_o):
        self.__delta_b_o = delta_b_o

    @property
    def W_f(self):
        return self.__W_f

    @W_f.setter
    def W_f(self, W_f):
        self.__W_f = W_f

    @property
    def U_f(self):
        return self.__U_f

    @U_f.setter
    def U_f(self, U_f):
        self.__U_f = U_f

    @property
    def b_f(self):
        return self.__b_f

    @b_f.setter
    def b_f(self, b_f):
        self.__b_f = b_f

    @property
    def delta_W_f(self):
        return self.__delta_W_f

    @delta_W_f.setter
    def delta_W_f(self, delta_W_f):
        self.__delta_W_f = delta_W_f

    @property
    def delta_U_f(self):
        return self.__delta_U_f

    @delta_U_f.setter
    def delta_U_f(self, delta_U_f):
        self.__delta_U_f = delta_U_f

    @property
    def delta_b_f(self):
        return self.__delta_b_f

    @delta_b_f.setter
    def delta_b_f(self, delta_b_f):
        self.__delta_b_f = delta_b_f

    @property
    def W_c(self):
        return self.__W_c

    @W_c.setter
    def W_c(self, W_c):
        self.__W_c = W_c

    @property
    def U_c(self):
        return self.__U_c

    @U_c.setter
    def U_c(self, U_c):
        self.__U_c = U_c

    @property
    def b_c(self):
        return self.__b_c

    @b_c.setter
    def b_c(self, b_c):
        self.__b_c = b_c

    @property
    def delta_W_c(self):
        return self.__delta_W_c

    @delta_W_c.setter
    def delta_W_c(self, delta_W_c):
        self.__delta_W_c = delta_W_c

    @property
    def delta_U_c(self):
        return self.__delta_U_c

    @delta_U_c.setter
    def delta_U_c(self, delta_U_c):
        self.__delta_U_c = delta_U_c

    @property
    def delta_b_c(self):
        return self.__delta_b_c

    @delta_b_c.setter
    def delta_b_c(self, delta_b_c):
        self.__delta_b_c = delta_b_c

    @property
    def params(self):
        if self.use_bias:
            return [self.W_i, self.U_i, self.b_i,
                    self.W_o, self.U_o, self.b_o,
                    self.W_f, self.U_f, self.b_f,
                    self.W_c, self.U_c, self.b_c]
        return [self.W_i, self.U_i,
                self.W_o, self.U_o,
                self.W_f, self.U_f,
                self.W_c, self.U_c]

    @property
    def grads(self):
        if self.use_bias:
            return [self.delta_W_i, self.delta_U_i, self.delta_b_i,
                    self.delta_W_o, self.delta_U_o, self.delta_b_o,
                    self.delta_W_f, self.delta_U_f, self.delta_b_f,
                    self.delta_W_c, self.delta_U_c, self.delta_b_c]
        return [self.delta_W_i, self.delta_U_i,
                self.delta_W_o, self.delta_U_o,
                self.delta_W_f, self.delta_U_f,
                self.delta_W_c, self.delta_U_c]

    def call(self, pre_layer=None, *args, **kwargs):
        self.connection(pre_layer)
        return self

    def connection(self, pre_layer):
        super(LSTM, self).connection(pre_layer)
        self.W_i = self.kernel_initializer((self.input_dim, self.output_dim))
        self.U_i = self.recurrent_initializer((self.output_dim, self.output_dim))
        self.b_i = np.zeros((self.output_dim,))

        self.W_o = self.kernel_initializer((self.input_dim, self.output_dim))
        self.U_o = self.recurrent_initializer((self.output_dim, self.output_dim))
        self.b_o = np.zeros((self.output_dim,))

        self.W_f = self.kernel_initializer((self.input_dim, self.output_dim))
        self.U_f = self.recurrent_initializer((self.output_dim, self.output_dim))
        self.b_f = np.zeros((self.output_dim,))

        self.W_c = self.kernel_initializer((self.input_dim, self.output_dim))
        self.U_c = self.recurrent_initializer((self.output_dim, self.output_dim))
        self.b_c = np.zeros((self.output_dim,))

    def forward(self, inputs, *args, **kwargs):
        # inputs: batch_size, time_step, out_dim
        inputs = np.asarray(inputs)
        self.inputs = inputs
        assert list(inputs.shape[1:]) == list(self.input_shape[1:])
        self.input_shape = inputs.shape
        self.output_shape[0] = self.input_shape[0]
        nb_batch, nb_seq, nb_input_dim = self.input_shape

        self.cells = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.outputs = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.logits_i = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.logits_o = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.logits_f = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.logits_c = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.outputs_i = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.outputs_o = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.outputs_f = np.zeros((nb_batch, nb_seq, self.output_dim))
        self.outputs_c = np.zeros((nb_batch, nb_seq, self.output_dim))
        for t in xrange(nb_seq):
            h_pre = np.zeros((nb_batch, self.output_dim)) \
                        if t == 0 else self.outputs[:, t - 1, :]
            c_pre = np.zeros((nb_batch, self.output_dim)) \
                        if t == 0 else self.cells[:, t - 1, :]
            x_now = self.inputs[:,t,:]
            i = x_now.dot(self.W_i) + h_pre.dot(self.U_i)
            o = x_now.dot(self.W_o) + h_pre.dot(self.U_o)
            f = x_now.dot(self.W_f) + h_pre.dot(self.U_f)
            c_tilde = x_now.dot(self.W_c) + h_pre.dot(self.U_c)

            if self.use_bias:
                i += self.b_i
                o += self.b_o
                f += self.b_f
                c_tilde += self.b_c

            self.logits_i[:,t,:] = i
            self.logits_o[:,t,:] = o
            self.logits_f[:,t,:] = f
            self.logits_c[:,t,:] = c_tilde

            i = self.recurrent_activator.forward(i)
            o = self.recurrent_activator.forward(o)
            f = self.recurrent_activator.forward(f)
            c_tilde = self.activator.forward(c_tilde)

            self.outputs_i[:,t,:] = i
            self.outputs_o[:,t,:] = o
            self.outputs_f[:,t,:] = f
            self.outputs_c[:,t,:] = c_tilde

            self.cells[:,t,:] = f * c_pre + i * c_tilde
            self.outputs[:,t,:] = o * self.activator.forward(self.cells[:,t,:])

        if self.return_sequences:
            return self.outputs
        else:
            return self.outputs[:,-1,:]

    def backward(self, pre_delta, *args, **kwargs):
        self.delta_W_i = np.zeros(self.W_i.shape)
        self.delta_W_o = np.zeros(self.W_o.shape)
        self.delta_W_f = np.zeros(self.W_f.shape)
        self.delta_W_c = np.zeros(self.W_c.shape)
        self.delta_U_i = np.zeros(self.U_i.shape)
        self.delta_U_o = np.zeros(self.U_o.shape)
        self.delta_U_f = np.zeros(self.U_f.shape)
        self.delta_U_c = np.zeros(self.U_c.shape)
        if self.use_bias:
            self.delta_b_i = np.zeros(self.b_i.shape)
            self.delta_b_o = np.zeros(self.b_o.shape)
            self.delta_b_f = np.zeros(self.b_f.shape)
            self.delta_b_c = np.zeros(self.b_c.shape)

        nb_batch, nb_seq, nb_input_dim = self.input_shape
        nb_output_dim = self.output_dim
        self.delta = np.zeros(self.input_shape)
        if self.return_sequences:
            assert len(pre_delta.shape) == 3
            # 此处的time_delta为delta_E/delta_output
            time_delta = pre_delta[:,nb_seq - 1,:]
        else:
            assert len(pre_delta.shape) == 2
            # 此处的time_delta为delta_E/delta_output
            time_delta = pre_delta
        future_state = np.zeros((nb_batch, nb_output_dim))
        for t in np.arange(nb_seq)[::-1]:
            logit_i = self.logits_i[:,t,:]
            logit_o = self.logits_o[:,t,:]
            logit_f = self.logits_f[:,t,:]
            logit_c_tilde = self.logits_c[:,t,:]
            output_i = self.outputs_i[:,t,:]
            output_o = self.outputs_o[:,t,:]
            output_c_tilde = self.outputs_c[:,t,:]
            c = self.cells[:,t,:]
            pre_c = np.zeros((nb_batch, nb_output_dim)) \
                        if t == 0 else self.cells[:,t - 1,:]
            pre_h = np.zeros((nb_batch, nb_output_dim)) \
                        if t == 0 else self.outputs[:,t - 1,:]
            cur_x = self.inputs[:,t,:]
            future_output_f = np.zeros((nb_batch, nb_output_dim)) \
                        if t == nb_seq - 1 else self.outputs_f[:,t + 1,:]
            # cell state
            pre_delta_state = time_delta * output_o * self.activator.backward(c) + \
                                future_state * future_output_f
            pre_delta_i = pre_delta_state * output_c_tilde * self.recurrent_activator.backward(logit_i)
            pre_delta_o = time_delta * self.activator.forward(c) * \
                          self.recurrent_activator.backward(logit_o)
            pre_delta_f = pre_delta_state * pre_c * self.recurrent_activator.backward(logit_f)
            pre_delta_c = pre_delta_state * output_i * self.activator.backward(logit_c_tilde)

            future_state = pre_delta_state

            # 求U的梯度
            self.delta_U_i += np.dot(pre_h.T, pre_delta_i) / nb_batch
            self.delta_U_o += np.dot(pre_h.T, pre_delta_o) / nb_batch
            self.delta_U_f += np.dot(pre_h.T, pre_delta_f) / nb_batch
            self.delta_U_c += np.dot(pre_h.T, pre_delta_c) / nb_batch
            # 求W的梯度
            self.delta_W_i += np.dot(cur_x.T, pre_delta_i) / nb_batch
            self.delta_W_o += np.dot(cur_x.T, pre_delta_o) / nb_batch
            self.delta_W_f += np.dot(cur_x.T, pre_delta_f) / nb_batch
            self.delta_W_c += np.dot(cur_x.T, pre_delta_c) / nb_batch
            # 求b的梯度
            if self.use_bias:
                self.delta_b_i += np.mean(pre_delta_i, axis=0)
                self.delta_b_o += np.mean(pre_delta_o, axis=0)
                self.delta_b_f += np.mean(pre_delta_f, axis=0)
                self.delta_b_c += np.mean(pre_delta_c, axis=0)
            # 求传到上一层的误差,layerwise
            self.delta[:,t,:] = np.dot(pre_delta_i, self.W_i.T) + \
                                np.dot(pre_delta_o, self.W_o.T) + \
                                np.dot(pre_delta_f, self.W_f.T) + \
                                np.dot(pre_delta_c, self.W_c.T)
            # 求同一层不同时间的误差,timewise
            if t > 0:
                time_delta = np.dot(pre_delta_i, self.U_i.T) + \
                             np.dot(pre_delta_o, self.U_o.T) + \
                             np.dot(pre_delta_f, self.U_f.T) + \
                             np.dot(pre_delta_c, self.U_c.T)
                if self.return_sequences:
                    time_delta += pre_delta[:,t - 1,:]
        return self.delta


class GRU(Recurrent):
    pass