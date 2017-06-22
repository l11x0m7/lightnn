# -*- encoding:utf-8 -*-


class Layer(object):
    def connection(self, pre_layer):
        raise NotImplementedError('function `connection` should be implemented')

    def params(self):
        raise NotImplementedError('function `get_params` should be implemented')

    def grads(self):
        raise NotImplementedError('function `get_grads` should be implemented')

    def forward(self, input):
        raise NotImplementedError('function `forward` should be implemented')

    def backward(self, pre_delta):
        raise NotImplementedError('function `backward` should be implemented')
