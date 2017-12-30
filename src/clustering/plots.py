#!/usr/bin/env python3

import matplotlib.pyplot as plt


def plot1(X):
    plt.scatter(X[:, 0], X[:, 1], s=50)
    plt.ylim(-2, 10)
    plt.xlim(-6, 6)
    plt.gca().set_aspect('equal')
    plt.show()


def plot2(X, labels):
    plt.scatter(X[:, 0], X[:, 1], s=50, c=labels, cmap='viridis')
    plt.ylim(-2, 10)
    plt.xlim(-6, 6)
    plt.gca().set_aspect('equal')
    plt.show()