#!/usr/bin/env python3

import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from utils import build_distance
from sklearn.cluster import KMeans as KMeansSK
from plots import plot1, plot2

np.random.seed(123)


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
        self.clusters = None

    def fit_predict(self, X):
        self.n_row, self.n_col = X.shape
        centroid_index = np.random.choice(self.n_row, size=self.k)
        centroids = X[centroid_index]

        for i in range(self.max_iter):
            distances_to_centroids = build_distance(X, centroids)
            cluster_assignment = np.argmin(distances_to_centroids, axis=1)
            new_centroids = np.array([X[cluster_assignment == i].mean(axis=0) for i in range(self.k)])

            if np.all(cluster_assignment == self.clusters):
                print('converged... iter {}'.format(i))
                break

            self.clusters = cluster_assignment
            centroids = new_centroids

        return self.clusters

    @property
    def inertia_(self):
        t_sse = []
        for i in range(self.k):
            cluster_data = X[self.clusters == i]
            cluster_mean = np.mean(cluster_data, axis=0)
            sse = np.sum(np.square(cluster_data - cluster_mean))
            t_sse.append(sse)

        return np.sum(t_sse)


if __name__ == '__main__':
    # Generate Data
    X, y = make_blobs(n_samples=300, centers=3,
                      random_state=0, cluster_std=0.6)

    # Plot Data
    # plot1(X)

    # K-Means Clustering
    kmeans = KMeans(k=3, max_iters=100)
    labels = kmeans.fit_predict(X)
    print(kmeans.inertia_)
    #plot2(X, labels)

    kmeans_sk = KMeansSK(n_clusters=3, max_iter=100, init='random')
    labels = kmeans_sk.fit_predict(X)
    plot2(X, labels)
    #print(kmeans_sk.inertia_)
