#!/usr/bin/env python3
from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.stats import itemfreq

class NBClassifier():

    def __init__(self):
        self.nrow = None
        self.ncol = None

    def get_priors(self, y):
        # P(y)

        item_freq = itemfreq(y)
        prior = dict([(x, y/self.nrow) for x, y in item_freq])
        return prior

    def get_likelihood(self):
        pass

    def fit(self, X, y):
        self.nrow, self.ncol = X.shape
        self.get_priors(y)

        exit()

        # P(x/y)
        likelyhood = np.zeros((2, self.ncol))
        for output in range(2):
            subset = X[np.equal(X, output)]
            print(subset)
            likelyhood[index, :] = np.sum(subset, axis=0) / subset.shape[0]

        print(likelyhood)


if __name__ == '__main__':
    X, y = make_classification(n_samples=300, n_features=4, n_classes=2, random_state=123)

    # Sklearn NB
    clf = BernoulliNB()
    clf.fit(X, y)
    y_hat = clf.predict(X)
    train_acc = accuracy_score(y, y_hat)

    clf = NBClassifier()
    clf.fit(X, y)
