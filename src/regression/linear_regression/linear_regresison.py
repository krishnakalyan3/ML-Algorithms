#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from plots import plot1, plot2

# w1x1 = y
# W * X = y
# W = [W1, W2, ..]
# X = [X1, X2, ..]

# W * X = y
# W * X * X.T =  X.T * y
# W =  (X.T * y) * (X.T * X ) ^ -1


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
    X, y = make_regression(n_samples=300, n_features=2, n_targets=1, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

    clf = LinearRegression()
    clf.fit(X_train, y_train)

    y_hat = clf.predict(X_test)
    print(mean_squared_error(y_test, y_hat))


# Disadvantages
# inverse operation is ND?
# invertability
# when is a matrix invertable
# There is not redundancy in data
# costly O(n^3)

# O(n^2)
