#!/usr/bin/env python3
import numpy as np

n_feature = 2
n_class = 2


def make_network(n_hidden=100):
    model = dict(
        W1=np.random.randn(n_feature, n_hidden),
        W2=np.random.randn(n_hidden, n_class)
    )
    return model
