#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.datasets.samples_generator import make_blobs


def euclidean_distance(X, center):
    """
    d(p,q) = sqrt((p1-q1)^2 + (p2-q2)^2 ... )
    """

    abs_dist = X - center
    sum_sq_abs_dist = np.sum(np.square(abs_dist), axis=1)
    sqrt_all = np.sqrt(sum_sq_abs_dist)
    return sqrt_all


def build_distance(X, centers):
    k = centers.shape[0]
    dist_matrix = np.zeros((X.shape[0], k))
    for i in range(k):
        dist_matrix[:, i] = euclidean_distance(X, centers[i])

    return dist_matrix


if __name__ == '__main__':
    X, y = make_blobs(n_samples=5, centers=3,
                      random_state=0, cluster_std=0.6)
    centers = np.array([[0.5, -0.5], [-2, 2]])

    pwd = pairwise_distances(X, centers, metric="euclidean")
    pwd_new = build_distance(X, centers)