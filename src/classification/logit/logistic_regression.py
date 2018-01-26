#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import make_classification
from scipy.special import expit as sigmoid
from sklearn.metrics import classification_report


class LogisticRgression():

    def __init__(self):
        self.weight = None

    def cost(self, X, y):
        yhat = sigmoid(X.dot(self.weight))
        error = y -yhat
        return error

    def sgd(self, X, y, num_iter=4000, lr=0.01):
        self.weight = np.random.rand(X.shape[1])
        for _ in range(num_iter):
            self.weight -= - lr * self.cost(X, y).dot(X)

    def fit(self, X, y):
        self.sgd(X,y)

    def classify(self, y):
        if y > 0.5:
            return 1
        else:
            return 0

    def predict(self, X):
        return np.array([self.classify(i) for i in  X.dot(self.weight)])


if __name__ == '__main__':
    X, y = make_classification(n_samples=300, n_features=4, n_classes=2, random_state=123)

    clf = LogisticRgression()
    clf.fit(X,y)
    yhat = clf.predict(X)

    print(classification_report(y, yhat))
