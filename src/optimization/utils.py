#!/usr/bin/env python3

import numpy as np
from sklearn.utils import shuffle


def forward(x, model):
    # Input to hidden
    h = x @ model['W1']
    h[h < 0] = 0

    # Hidden to output
    prob = softmax(h @ model['W2'])
    return h, prob


def backward(model, xs, hs, errs):
    dW2 = hs.T @ errs
    dh = errs @ model['W2'].T

    # relu
    dh[hs < 0] = 0
    dW1 = xs.T @ dh
    return dict(W1=dW1, W2=dW2)


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


def get_minibatch(X, y, batch_size=100):
    mini_batches = []

    X, y = shuffle(X, y)
    for i in range(0, X.shape[0], batch_size):
        x_mini = X[i: (i + batch_size)]
        y_mini = y[i: (i + batch_size)]
        mini_batches.append((x_mini, y_mini))

    return mini_batches


def get_minibatch_grad(model, X_train, y_train):
    xs, hs, errs = [], [], []

    for x, cls_idx in zip(X_train, y_train):
        h, y_pred = forward(x, model)

        y_true = np.zeros(2)
        y_true[int(cls_idx)] = 1.
        err = y_true - y_pred

        xs.append(x)
        hs.append(h)
        errs.append(err)

    return backward(model, np.array(xs), np.array(hs), np.array(errs))


def accuracy(y, y_hat):
    accuracy = 0
    for y, y_pred in zip(y, y_hat):
        if y == y_pred:
            accuracy += 1

    return accuracy/len(y_hat)


def predict(model, x_test, y_test):
    y_pred = np.zeros_like(y_test)

    for i, x in enumerate(x_test):
        _, prob = forward(x, model)
        y = np.argmax(prob)
        y_pred[i] = y
    return y_pred
