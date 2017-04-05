#-*- encoding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
from ..base.BasicFunctions import Sigmoid
from ..base.Costs import CECost
from ..base.Initializers import xavier_weight_initializer
from ..layers.FullyConnectedLayer import FullyConnectedLayer


class NetWork(object):
    def __init__(self, sizes, cost=CECost, activator=Sigmoid, initializer=xavier_weight_initializer, lr=1e-1, lmbda=None):
        self.sizes = sizes
        self.layer_num = len(sizes)
        self.cost = cost
        self.lr = lr
        self.lmbda = lmbda
        self.layers = [FullyConnectedLayer(i_size, o_size,
                            activator, initializer) for i_size, o_size
                                in zip(sizes[:-1], sizes[1:])]

    def train(self, input_data, input_label, epoch, batch_size, verbose=True):
        for ep in xrange(epoch):
            batch_epoch = len(input_data) // batch_size + 1
            for mini_batch, batch_label in self.__batch_sample(batch_epoch, batch_size, input_data, input_label):
                self.train_batch(mini_batch, batch_label, len(input_data))

            if verbose:
                print("Epoch %s training complete" % ep)
                cost = self.total_cost(input_data, input_label)
                print("Cost on training data: {}".format(cost))
                accuracy = self.accuracy(input_data, input_label)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, len(input_data)))

    def train_batch(self, mini_batch, batch_label, training_set_n):
        batch_delta_W = [np.zeros(layer.get_W().shape) for layer in self.layers]
        batch_delta_b = [np.zeros(layer.get_b().shape) for layer in self.layers]
        for x, y in zip(mini_batch, batch_label):
            self.feedforward(x)
            self.backprop(y)
            for layer_num in xrange(self.layer_num-1):
                batch_delta_W[layer_num] += self.layers[layer_num].delta_W
                batch_delta_b[layer_num] += self.layers[layer_num].delta_b


        # print batch_delta_W
        for i, layer in enumerate(self.layers):
            update_delta_W = batch_delta_W[i] * self.lr / len(mini_batch)
            update_delta_b = batch_delta_b[i] * self.lr / len(mini_batch)
            if self.lmbda is not None:
                update_delta_W += self.lmbda * self.lr * layer.get_W() / training_set_n
            layer.step(update_delta_W, update_delta_b)


    def backprop(self, y):
        """
        Calculate the gradient of each parameter
        :param x: single input data
        :param y: single one hot output label
        :return: None
        """
        delta = self.cost.delta(self.layers[-1].logit, self.layers[-1].output, y)
        self.layers[-1].update_delta(delta)
        for l in xrange(2, self.layer_num):
            delta = self.layers[-l].backward(pre_delta=delta, pre_W=self.layers[-l+1].W)


    def feedforward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def __batch_sample(self, batch_epoch, batch_size, input_data, input_label):
        for _ in xrange(batch_epoch):
            choice = np.random.choice(len(input_data), batch_size, replace=False)
            yield np.asarray(input_data)[choice], np.asarray(input_label)[choice]


    def accuracy(self, data, label):
        results = 0
        for x, y in zip(data, label):
            self.feedforward(x)
            a = self.layers[-1].output
            results += (np.argmax(a) == np.argmax(y))
        return results


    def total_cost(self, data, label):
        cost = 0.0
        for x, y in zip(data, label):
            self.feedforward(x)
            a = self.layers[-1].output
            cost += self.cost.cost(a, y) / len(data)
        if self.lmbda is not None:
            cost += 0.5 * (self.lmbda / len(data)) * sum(
                np.linalg.norm(layer.get_W())**2 for layer in self.layers)
        return cost



