#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import itemfreq


class KNNClassifier():

    def __init__(self, n_neighbors=5, p=2):
        """
        Nearest Neighbour implementation

        Parameters
        ----------
        n_neighbors : Integer,  default 5
        The number of neighbors to take into account

        p : function, euclidean distance
        Distance function between two arguments
        """
        self.n_neighbors = n_neighbors
        self.p = p

    def most_common(self, y):
        y_freq = itemfreq(y)
        sort_freq = sorted(y_freq, key=lambda k: k[1])
        return sort_freq[-1][0]

    def euclidean_distance(self, X, x):
        """
        d(p,q) = sqrt((p1-q1)^2 + (p2-q2)^2 ... )
        """
        abs_dist = X - x
        sum_sq_abs_dist = np.sum(np.square(abs_dist), axis=1)
        sqrt_all = np.sqrt(sum_sq_abs_dist)
        return sqrt_all

    def fit_predict(self, X, y):
        y_hat = []
        for x in X:
            if self.p == 2:
                distances = self.euclidean_distance(X, x)
            neighbors_distance = sorted((distance, target) for distance, target in zip(distances, y))
            top_n = neighbors_distance[:self.n_neighbors]
            neighbors_targets = self.most_common([y for _, y in top_n])
            y_hat.append(neighbors_targets)
        return y_hat


if __name__ == '__main__':
    X, y = make_classification(n_samples=300, n_features=4, n_classes=2, random_state=123)

    clf1 = KNNClassifier()
    y_hat = clf1.fit_predict(X, y)
    print(y_hat)
    train_acc = accuracy_score(y, y_hat)
    print(train_acc)

    clf = KNeighborsClassifier(n_neighbors=3, p=2)
    clf.fit(X, y)
    y_hat = clf.predict(X)

    train_acc = accuracy_score(y, y_hat)
    print(train_acc)
