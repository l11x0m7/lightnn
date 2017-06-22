# -*- encoding:utf-8 -*-

import numpy as np


class Optimizer(object):
    def __init__(self, lr=1e-3, decay=0., grad_clip=-1, lr_min=0., lr_max=np.inf):
        self.lr = lr
        self.decay = decay
        self.clip = grad_clip
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.iterations = 0

    def update(self):
        self.iterations += 1
        self.lr *= (1. / 1 + self.decay * self.iterations)
        self.lr = np.clip(self.lr, self.lr_min, self.lr_max)


class SGD(Optimizer):
    def __init__(self, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)

    def minimize(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * _grad_clip(g, self.clip)
        super(SGD, self).update()

    def maximum(self, params, grads):
        for p, g in zip(params, grads):
            p += self.lr * _grad_clip(g, self.clip)
        super(SGD, self).update()


def _grad_clip(grad, clip):
    if clip > 0:
        return np.clip(grad, -clip, clip)
    else:
        return grad


def get(optimizer):
    if isinstance(optimizer, str):
        if optimizer in ('SGD', 'sgd'):
            return SGD()
        else:
            raise ValueError('Unknown optimizer name `{}`'.format(optimizer))
    elif isinstance(optimizer, Optimizer):
        return optimizer
    else:
        raise ValueError('Unknown optimizer type `{}`'.format(optimizer.__class__.__name__))
