#!/usr/bin/env python3


class BaseEstimator(object):
    X = None
    y = None
    y_required = True
    fit_required = True

    def _setup_input(self, X, y=None):
        """
        Ensures that the inputs are in the required formats.
        """

        if X.size == 0:
            raise ValueError('Number of features must be > 0')

        if self.y_required:

            if y is None:
                raise ValueError('Missed required argument y')

            if y.size == 0:
                raise ValueError('Number of targets must be > 0')
