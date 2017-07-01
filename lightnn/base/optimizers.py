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


class Momentum(Optimizer):
    """
        Performs stochastic gradient descent with momentum.

        momentum: Scalar between 0 and 1 giving the momentum value.
            Setting momentum = 0 reduces to sgd.
        velocity: A numpy array of the same shape as w and dw used to store a moving
            average of the gradients.
    """
    def __init__(self, momentum=0.9, *args, **kwargs):
        super(Momentum, self).__init__(*args, **kwargs)
        self.momentum = momentum
        self.velocity = dict()

    def minimize(self, params, grads):
        for p, g in zip(params, grads):
            v = self.velocity.get(id(p), np.zeros_like(p))
            v = self.momentum * v - self.lr * g
            p += v
            self.velocity[id(p)] = v
        super(Momentum, self).update()

    def maximum(self, params, grads):
        for p, g in zip(params, grads):
            v = self.velocity.get(id(p), np.zeros_like(p))
            v = self.momentum * v - self.lr * g
            p -= v
            self.velocity[id(p)] = p
        super(Momentum, self).update()


class RMSProp(Optimizer):
    """
        Uses the RMSProp update rule, which uses a moving average of squared gradient
        values to set adaptive per-parameter learning rates.

    learning_rate: Scalar learning rate.
    decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
        gradient cache.
    epsilon: Small scalar used for smoothing to avoid dividing by zero.
    cache: Moving average of second moments of gradients.
    """
    def __init__(self, decay_rate=0.99, epsilon=1e-8, *args, **kwargs):
        super(RMSProp, self).__init__(*args, **kwargs)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = dict()

    def minimize(self, params, grads):
        for p, g in zip(params, grads):
            cache = self.cache.get(id(p), np.zeros_like(p))
            self.cache[id(p)] = self.decay_rate * cache + (1 - self.decay_rate) * (g ** 2)
            p -= self.lr * g / (np.sqrt(self.cache[id(p)]) + self.epsilon)
        super(RMSProp, self).update()

    def maximum(self, params, grads):
        for p, g in zip(params, grads):
            cache = self.cache.get(id(p), np.zeros_like(p))
            self.cache[id(p)] = self.decay_rate * cache + (1 - self.decay_rate) * (g ** 2)
            p += self.lr * g / (np.sqrt(self.cache[id(p)]) + self.epsilon)
        super(RMSProp, self).update()


class Adam(Optimizer):
    """
        Uses the Adam update rule, which incorporates moving averages of both the
        gradient and its square and a bias correction term.

        beta1: Decay rate for moving average of first moment of gradient.
        beta2: Decay rate for moving average of second moment of gradient.
        epsilon: Small scalar used for smoothing to avoid dividing by zero.
        m: Moving average of gradient.
        v: Moving average of squared gradient.
    """
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, *args, **kwargs):
        super(Adam, self).__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = dict()
        self.v = dict()

    def minimize(self, params, grads):
        for p, g in zip(params, grads):
            m = self.m.get(id(p), np.zeros_like(p))
            v = self.v.get(id(p), np.zeros_like(p))
            self.m[id(p)] = self.beta1 * m + (1 - self.beta1) * g
            self.v[id(p)] = self.beta2 * v + (1 - self.beta2) * g ** 2
            mb = self.m[id(p)] / (1 - self.beta1 ** (self.iterations + 1))
            vb = self.v[id(p)] / (1 - self.beta2 ** (self.iterations + 1))
            p -= (self.lr * mb / (np.sqrt(vb) + self.epsilon))
        super(Adam, self).update()

    def maximum(self, params, grads):
        for p, g in zip(params, grads):
            m = self.m.get(id(p), np.zeros_like(p))
            v = self.v.get(id(p), np.zeros_like(p))
            self.m[id(p)] = self.beta1 * m + (1 - self.beta1) * g
            self.v[id(p)] = self.beta2 * v + (1 - self.beta2) * g
            mb = self.m[id(p)] / (1 - self.beta1 ** (self.iterations + 1))
            vb = self.v[id(p)] / (1 - self.beta2 ** (self.iterations + 1))
            p += (self.lr * mb / (np.sqrt(vb) + self.epsilon))
        super(Adam, self).update()


class Adagrad(Optimizer):
    def __init__(self, epsilon=1e-7, *args, **kwargs):
        super(Adagrad, self).__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.r = dict()

    def minimize(self, params, grads):
        for p, g in zip(params, grads):
            self.r.setdefault(id(p), np.zeros_like(p))
            self.r[id(p)] += g ** 2
            p -= self.lr / (self.epsilon + np.sqrt(self.r[id(p)])) * g
        super(Adagrad, self).update()

    def maximum(self, params, grads):
        for p, g in zip(params, grads):
            self.r.setdefault(id(p), np.zeros_like(p))
            self.r[id(p)] += g ** 2
            p += self.lr / (self.epsilon + np.sqrt(self.r[id(p)])) * g
        super(Adagrad, self).update()


def _grad_clip(grad, clip):
    if clip > 0:
        return np.clip(grad, -clip, clip)
    else:
        return grad


def get(optimizer):
    if isinstance(optimizer, str):
        optimizer = optimizer.lower()
        if optimizer in ('sgd', ):
            return SGD()
        elif optimizer in ('momentum', ):
            return Momentum()
        elif optimizer in ('rmsprop', 'rms'):
            return RMSProp()
        elif optimizer in ('adam'):
            return Adam()
        elif optimizer in ('adagrad', ):
            return Adagrad()
        else:
            raise ValueError('Unknown optimizer name `{}`'.format(optimizer))
    elif isinstance(optimizer, Optimizer):
        return optimizer
    else:
        raise ValueError('Unknown optimizer type `{}`'.format(optimizer.__class__.__name__))
