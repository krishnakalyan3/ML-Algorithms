#!/usr/bin/env python3

import numpy as np
from scipy.linalg import svd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from base import BaseEstimator
import logging

np.random.seed(1337)


class PCA(BaseEstimator):
    y_required = False

    def __init__(self, n_components, solver='svd'):
        """
        Principal component analysis (PCA) implementation.

        Parameters
        ----------
        n_components : Integer
        Number of Components

        solver : String
        Optional arguments {'svd', 'eigen'}
        """

        self.solver = solver
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self._scale(X)
        self._decompose(X)

    def transform(self, X):
        X -= self.mean
        return np.dot(X, self.components.T)

    def _scale(self, X):
        mean = np.mean(X, axis=0)
        self.mean = mean

    def _decompose(self, X):
        X -= self.mean

        if self.solver == 'svd':
            U, s, V = svd(X)

        if self.solver == 'eigen':
            cov_mat = np.cov(X.T)
            s, V = np.linalg.eig(cov_mat)

        self._variance_explained(s)
        self.components = V[0:self.n_components]

    def _variance_explained(self, s):
        s_squared = s ** 2
        variance_ratio = s_squared / (s_squared).sum()
        print('Explained variance ratio: %s' % (variance_ratio[0:self.n_components]))


if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=75,
                               random_state=1111, n_classes=2, class_sep=2.5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=1111)

    pca = PCA(n_components=10, solver='svd')
    pca.fit(X_train)

