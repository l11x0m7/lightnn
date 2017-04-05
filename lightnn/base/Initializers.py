# -*- encoding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
def xavier_weight_initializer(shape):
    """Defines an initializer for the Xavier distribution.

    This function will be used as a variable scope initializer.

    https://www.tensorflow.org/versions/r0.7/how_tos/variable_scope/index.html#initializers-in-variable-scope

    Args:
    shape: Tuple or 1-d array that species dimensions of requested tensor.
    Returns:
    out: tf.Tensor of specified shape sampled from Xavier distribution.
    """
    m = shape[0]
    n = shape[1] if len(shape)>1 else shape[0]
    bound = np.sqrt(6. / (m + n))
    out = np.random.uniform(-bound, bound, shape)
    return out

def default_weight_initializer(shape):
    return np.random.randn(shape) / np.sqrt(shape[1])

def large_weight_initializer(shape):
    return np.random.randn(shape)
