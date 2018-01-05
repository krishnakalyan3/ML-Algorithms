#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from model import make_network
from utils import get_minibatch, get_minibatch_grad, forward, accuracy, predict


def sgd(model, x_train, y_train):
    mini_batches = get_minibatch(x_train, y_train)

    for i in range(n_iter):
        idx = np.random.randint(0, len(mini_batches))
        x_mini, y_mini = mini_batches[idx]

        grad = get_minibatch_grad(model, x_mini, y_mini)
        for layer in grad:
            model[layer] += 1e-3 * grad[layer]

    return model


def momentum(model, X_train, y_train):
    velocity = {k: np.zeros_like(v) for k, v in model.items()}
    gamma = .9

    minibatches = get_minibatch(X_train, y_train)

    for iter in range(1, 100 + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)

        for layer in grad:
            velocity[layer] = gamma * velocity[layer] + 1e-3 * grad[layer]
            model[layer] += velocity[layer]

    return model


if __name__ == '__main__':
    X, y = make_moons(n_samples=5000, random_state=42, noise=0.1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=123)

    mean_accuracy = []

    for j in range(15):
        model = make_network()
        model = sgd(model, x_train, y_train)
        y_pred = predict(model, x_test, y_test)

        acc = accuracy(y_test, y_pred)
        mean_accuracy.append(acc)

    print(np.mean(mean_accuracy))
