#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from base import BaseEstimator


class KNNClassifier(BaseEstimator):

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

