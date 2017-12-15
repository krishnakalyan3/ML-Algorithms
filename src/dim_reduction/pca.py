#!/usr/bin/env python3

import numpy as np
from scipy.linalg import svd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


np.random.seed(1337)


class PCA():
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
        self.scale = None
        self.var_explained = None

    def fit(self, X):
        self._scale(X)
        self._decompose(X)

    def transform(self, X):
        X -= self.mean
        return np.dot(X, self.components.T)

    def _scale(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)

    def _decompose(self, X):
        """
        S: Eigen Values
        V: Eigen Vectors
        """

        if self.solver == 'svd':
            X -= self.mean
            X /= self.scale
            U, s, V = svd(X)

        if self.solver == 'eigen':
            cov_mat = np.cov(X.T)
            s, V = np.linalg.eig(cov_mat)

        self.components = V[0:self.n_components]
        self.var_explained = self._variance_explained(s)

    def _variance_explained(self, s):
        variance_ratio = s / np.sum(s)
        return variance_ratio[:self.n_components]


if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=100, n_informative=75,
                               random_state=1111, n_classes=2, class_sep=2.5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=1111)

    #pca = PCA(n_components=10, solver='svd')
    #pca.fit(X_train)

    pca1 = PCA(n_components=10, solver='eigen')
    pca1.fit(X_train)
    print(pca1.var_explained)

    pca2 = PCA(n_components=10, solver='svd')
    pca2.fit(X_train)
    print(pca2.var_explained)