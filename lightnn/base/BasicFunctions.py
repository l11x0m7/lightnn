# -*- encoding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# --- sigmoid functions ---*

def sigmoid(z):
    return 1. / (1 + np.exp(-z))


def delta_sigmoid(z):
    return sigmoid(z) * (1. - sigmoid(z))

class Sigmoid(object):
    @staticmethod
    def forward(z):
        return sigmoid(z)
    @staticmethod
    def backward(z):
        return delta_sigmoid(z)


# --- relu functions ---*

def relu(z):
    z = np.asarray(z)
    return np.maximum(z, 0)

def delta_relu(z):
    z = np.asarray(z)
    return (z > 0).astype(int)


class Relu(object):
    @staticmethod
    def forward(z):
        return relu(z)
    @staticmethod
    def backward(z):
        return delta_relu(z)

# --- identity functions ---*

def identity(z):
    z = np.asarray(z)
    return z

def delta_identity(z):
    z = np.asarray(z)
    return np.ones(z.shape)


class Identity(object):
    @staticmethod
    def forward(z):
        return identity(z)

    @staticmethod
    def backward(z):
        return delta_identity(z)

# --- softmax functions ---*

def softmax(x):
    x = np.asarray(x)
    if len(x.shape) > 1:
        x -= x.max(axis=1).reshape([x.shape[0], 1])
        x = np.exp(x)
        x /= np.sum(x, axis=1).reshape([x.shape[0], 1])
        return x
    else:
        x -= np.max(x)
        x = np.exp(x)
        x /= np.sum(x)
        return x
