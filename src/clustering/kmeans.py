#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from utils import build_distance

np.random.seed(123)


def make_plot(X):
    plt.scatter(X[:, 0], X[:, 1], s=50)
    plt.ylim(-2, 10)
    plt.xlim(-6, 6)
    plt.gca().set_aspect('equal')
    plt.show()


class KMeans():
    """
    KMeans implementation.

    Parameters
    ----------
    k : int
    Number of Clusters

    max_iter : int
    Max number of iterations
    """

    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iter = max_iters
        self.n_row = None
        self.n_col = None
        self.centroid_index = None

    def fit(self, X):
        self.n_row, self.n_col = X.shape
        centroid_index = np.random.choice(self.n_row, size=self.k)
        centroids = X[centroid_index]

        distances_to_centroids = build_distance(X, centroids)
        print(m[0:3])


    def predict(self):
        pass



if __name__ == '__main__':
    X, y = make_blobs(n_samples=300, centers=3,
                      random_state=0, cluster_std=0.6)

    #make_plot(X)

    kmeans = KMeans()
    kmeans.fit(X)