# -*- encoding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from BasicFunctions import sigmoid, delta_sigmoid
import numpy as np

class QuadraticCost(object):
    """
    calculate the RMSE
    """
    @staticmethod
    def cost(y_hat, y, norm=2):
        """
        the default vector norm is : sum(abs(x)**2)**(1./2)
        :param y_hat: vector, output from your network
        :param y: vector, the ground truth
        :return: scalar, the cost
        """
        return 0.5 * np.linalg.norm(y_hat - y, ord=norm)

    @staticmethod
    def delta(z, y_hat, y, activator='sigmoid'):
        if activator == 'sigmoid':
            return (y_hat - y) * delta_sigmoid(z)
        else:
            raise ValueError('The activator the package support only includes '
                             'sigmoid so far!')


class CECost(object):
    @staticmethod
    def cost(y_hat, y):
        return np.sum(np.nan_to_num(- y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)))

    @staticmethod
    def delta(z, y_hat, y, activator='sigmoid'):
        """
        :param z:
        :param y_hat:
        :param y:
        :param activator:
        :return:
        """
        if activator == 'sigmoid':
            return y_hat - y
        else:
            raise ValueError('You must make sure the last layer uses sigmoid activator!')

class LogLikelihoodCost(object):
    @staticmethod
    def cost(y_hat, y):
        if np.sum(y_hat) != 1. or np.sum(y) != 1.:
            raise ValueError('THe y_hat and y must be probability!')
        return np.sum(np.nan_to_num(- y * np.log(y_hat)))

    @staticmethod
    def delta(z, y_hat, y, activator='softmax'):
        """
        The loss partial by z is : y_hat * (y - y_hat) / (-1 / y_hat) = y_hat - y
        softmax + loglikelihoodCost == sigmoid + crossentropyCost
        :param z:
        :param y_hat:
        :param y:
        :param activator:
        :return:
        """
        if activator == 'softmax' and np.sum(y_hat) == 1 and np.sum(y) == 1:
            return y_hat - y
        else:
            raise ValueError('You must use the softmax layer first!')