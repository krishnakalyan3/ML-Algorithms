#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class KNNClassifier():

    def __init__(self, k=5, dist_func=euclidean):
        """
        Nearest Neighbour implementation

        Parameters
        ----------
        k : Integer,  default 5
        The number of neighbors to take into account

        dist_func : function, euclidean distance
        Distance function between two arguments
        """
        self.k = k
        self.dist_func = dist_func


if __name__ == '__main__':
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5,
                               n_redundant=0, n_repeated=0, n_classes=3,
                               random_state=1111, class_sep=1.5, )


