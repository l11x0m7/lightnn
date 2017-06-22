# -*- encoding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from lightnn.models.models import Sequential
from lightnn.layers.core import Dense, Flatten, Softmax
from lightnn.layers.convolutional import Conv2d
from lightnn.layers.pooling import MaxPooling, AvgPooling
from lightnn.base.activations import Relu
from lightnn.base.optimizers import SGD


def mlp_random():
    input_size = 600
    input_dim = 20
    label_size = 10
    train_X = np.random.random((input_size, input_dim))
    train_y = np.zeros((input_size, label_size))
    for _ in xrange(input_size):
        train_y[_,np.random.randint(0, label_size)] = 1
    model = Sequential()
    model.add(Dense(100, input_size=input_dim))
    model.add(Softmax(label_size))
    model.compile('CCE')
    model.fit(train_X, train_y, verbose=1)


def mlp_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
    training_data = np.array([image.flatten() for image in mnist.train.images])
    training_label = mnist.train.labels
    valid_data = np.array([image.flatten() for image in mnist.validation.images])
    valid_label = mnist.validation.labels
    input_dim = training_data.shape[1]
    label_size = training_label.shape[1]
    model = Sequential()
    model.add(Dense(300, input_size=input_dim, activator=Relu))
    model.add(Softmax(label_size))
    model.compile('CCE', optimizer=SGD())
    model.fit(training_data, training_label, validation_data=(valid_data, valid_label))


def cnn_random():
    input_size = 600
    input_dim = 28
    input_depth = 1
    label_size = 10
    train_X = np.random.random((input_size, input_dim, input_dim, input_depth))
    train_y = np.zeros((input_size, label_size))
    for _ in xrange(input_size):
        train_y[_,np.random.randint(0, label_size)] = 1
    model =Sequential()
    model.add(Conv2d((3, 3), 1, input_shape=(None, 28, 28, 1),activator=Relu))
    model.add(AvgPooling((2, 2), stride=2))
    model.add(Conv2d((4, 4), 2, activator=Relu))
    model.add(AvgPooling((2, 2), stride=2))
    model.add(Flatten())
    model.add(Softmax(label_size))
    model.compile('CCE', optimizer=SGD(1e-2))
    model.fit(train_X, train_y)


def cnn_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
    training_data = np.array([image.reshape(28, 28, 1) for image in mnist.train.images])
    training_label = mnist.train.labels
    valid_data = np.array([image.reshape(28, 28, 1) for image in mnist.validation.images])
    valid_label = mnist.validation.labels
    label_size = training_label.shape[1]
    model =Sequential()
    model.add(Conv2d((3, 3), 1, input_shape=(None, 28, 28, 1),activator=Relu))
    model.add(AvgPooling((2, 2), stride=2))
    model.add(Conv2d((4, 4), 2, activator=Relu))
    model.add(AvgPooling((2, 2), stride=2))
    model.add(Flatten())
    model.add(Softmax(label_size))
    model.compile('CCE', optimizer=SGD())
    model.fit(training_data, training_label, validation_data=(valid_data, valid_label), verbose=2)


if __name__ == '__main__':
    # mlp_mnist()
    # cnn_mnist()
    cnn_random()
