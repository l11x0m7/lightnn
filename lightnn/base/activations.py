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

def softmax(z):
    z = np.asarray(z)
    if len(z.shape) > 1:
        z -= z.max(axis=1).reshape([z.shape[0], 1])
        z = np.exp(z)
        z /= np.sum(z, axis=1).reshape([z.shape[0], 1])
        return z
    else:
        z -= np.max(z)
        z = np.exp(z)
        z /= np.sum(z)
        return z

def delta_softmax(z):
    return np.ones(z.shape, dtype=z.dtype)


class Softmax(object):
    @staticmethod
    def forward(z):
        return softmax(z)

    @staticmethod
    def backward(z):
        return delta_softmax(z)