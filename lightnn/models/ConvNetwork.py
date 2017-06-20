#-*- encoding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np

from ..base.BasicFunctions import Sigmoid
from ..base.Costs import CECost
from ..base.Initializers import xavier_weight_initializer
from ..layers.ConvLayer import ConvLayer
from ..layers.PoolingLayer import MaxPoolingLayer, AvgPoolingLayer


class ConvNetwork(object):
    def __init__(self):
        pass