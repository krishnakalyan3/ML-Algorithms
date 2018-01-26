#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from plots import plot1, plot2


class LinearRegression():
    # TODO
    # implement gradient descent

    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        inverse = np.linalg.inv(X.T.dot(X))
        self.weights = inverse.dot(X.T).dot(y)

    def predict(self, X):
        return X.dot(self.weights)


if __name__ == '__main__':
    X, y = make_regression(n_samples=300, n_features=1, n_targets=1, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    #plot1(X)
    #plot2(X, y)

    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)

    #plot2(X, y_hat)
    print(mean_squared_error(y_test, y_hat))
