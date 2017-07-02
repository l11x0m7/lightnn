# -*- encoding:utf-8 -*-
import numpy as np

from lightnn.layers.recurrent import SimpleRNN


def vector_rnn_gradient_check():
    batch_size = 1
    time_step = 10
    out_dim = 15
    in_dim = 20
    use_bias = True

    data = np.random.random((batch_size, time_step, in_dim))

    sr = SimpleRNN(out_dim, (batch_size, time_step, in_dim), activator='leaky_relu', use_bias=use_bias)

    loss = lambda p: np.sum(p)
    pre_delta = np.ones((batch_size, out_dim))

    # check real grad
    y_hat = sr.forward(data)
    sr.backward(pre_delta)

    epsilon = 1e-5

    # check W
    for i in xrange(in_dim):
        for j in xrange(out_dim):
            bp_grad = sr.delta_W[i,j]
            sr.W[i,j] -= epsilon
            sr.reset()
            y_h1 = sr.forward(data)
            loss1 = loss(y_h1)
            sr.W[i,j] += 2 * epsilon
            sr.reset()
            y_h2 = sr.forward(data)
            loss2 = loss(y_h2)
            real_grad = (loss2 - loss1) / (2 * epsilon)
            print('W[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

    # check U
    for i in xrange(out_dim):
        for j in xrange(out_dim):
            bp_grad = sr.delta_U[i,j]
            sr.U[i,j] -= epsilon
            sr.reset()
            y_h1 = sr.forward(data)
            loss1 = loss(y_h1)
            sr.U[i,j] += 2 * epsilon
            sr.reset()
            y_h2 = sr.forward(data)
            loss2 = loss(y_h2)
            real_grad = (loss2 - loss1) / (2 * epsilon)
            print('U[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

    # check b
    if use_bias:
        for i in xrange(out_dim):
            bp_grad = sr.delta_b[i]
            sr.b[i] -= epsilon
            sr.reset()
            y_h1 = sr.forward(data)
            loss1 = loss(y_h1)
            sr.b[i] += 2 * epsilon
            sr.reset()
            y_h2 = sr.forward(data)
            loss2 = loss(y_h2)
            real_grad = (loss2 - loss1) / (2 * epsilon)
            print('b[{}],real grad:{}, bp grad:{}'.format(i, real_grad, bp_grad))


def sequence_rnn_gradient_check():
    batch_size = 200
    time_step = 10
    out_dim = 15
    out_dim2 = 5
    in_dim = 20
    use_bias = True

    data = np.random.random((batch_size, time_step, in_dim))

    sr1 = SimpleRNN(out_dim, (batch_size, time_step, in_dim),
                    activator='leaky_relu', use_bias=use_bias, return_sequences=True)
    sr2 = SimpleRNN(out_dim2)(sr1)

    loss = lambda p: np.sum(p)
    pre_delta = np.ones((batch_size, out_dim2))

    # check real grad
    y1_hat = sr1.forward(data)
    y2_hat = sr2.forward(y1_hat)
    delta2 = sr2.backward(pre_delta)
    delta1 = sr1.backward(delta2)

    epsilon = 1e-5

    # check W
    for i in xrange(in_dim):
        for j in xrange(out_dim):
            bp_grad = sr1.delta_W[i,j]
            sr1.W[i,j] -= epsilon
            sr1.reset()
            sr2.reset()
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss1 = loss(y2_hat)
            sr1.W[i,j] += 2 * epsilon
            sr1.reset()
            sr2.reset()
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss2 = loss(y2_hat)
            real_grad = (loss2 - loss1) / (2 * epsilon)
            print('W[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

    # check U
    for i in xrange(out_dim):
        for j in xrange(out_dim):
            bp_grad = sr1.delta_U[i,j]
            sr1.U[i,j] -= epsilon
            sr1.reset()
            sr2.reset()
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss1 = loss(y2_hat)
            sr1.U[i,j] += 2 * epsilon
            sr1.reset()
            sr2.reset()
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss2 = loss(y2_hat)
            real_grad = (loss2 - loss1) / (2 * epsilon)
            print('U[{},{}],real grad:{}, bp grad:{}'.format(i, j, real_grad, bp_grad))

    # check b
    if use_bias:
        for i in xrange(out_dim):
            bp_grad = sr1.delta_b[i]
            sr1.b[i] -= epsilon
            sr1.reset()
            sr2.reset()
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss1 = loss(y2_hat)
            sr1.b[i] += 2 * epsilon
            sr1.reset()
            sr2.reset()
            y1_hat = sr1.forward(data)
            y2_hat = sr2.forward(y1_hat)
            delta2 = sr2.backward(pre_delta)
            delta1 = sr1.backward(delta2)
            loss2 = loss(y2_hat)
            real_grad = (loss2 - loss1) / (2 * epsilon)
            print('b[{}],real grad:{}, bp grad:{}'.format(i, real_grad, bp_grad))


if __name__ == '__main__':
    vector_rnn_gradient_check()
    sequence_rnn_gradient_check()