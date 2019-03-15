"""
Useful functions and classes to process information in gym_hyq environments
"""

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class HyQStateScaler(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.n_in = 0
        self.mins = np.array([-1, -100000, -100000, -100000, -100000,
                              -0.3, 0, -2, -0.3, 0, -2, -0.3, -1, 0, -0.3, -1, 0,
                              -1.6, -3, -1.6, -1000, -1000, 0])
        self.maxs = np.array([1, 100000, 100000, 100000, 100000,
                              0.1, 1, 0, 0.1, 1, 0, 0.1, 0, 2, 0.1, 0, 2,
                              1.6, 3, 1.6, 1000, 1000, 1])
        self.x_scaled = None

    def _fit_transform(self, x):

        if isinstance(x, list):
            x = np.array(x)
        self.n_in = x.shape[0]

        x_std = (x - self.mins) / (self.maxs - self.mins)
        self.x_scaled = x_std * 2 - np.ones(x_std.shape)

        return self

    def fit(self, x):

        self = self._fit_transform(x)
        return self

    def fit_transform(self, x, y=None, **kwargs):

        self = self._fit_transform(x)
        return self.x_scaled

    def transform(self, x):

        self = self._fit_transform(x)
        return self.x_scaled

    def inverse_transform(self, x):

        if isinstance(x, list):
            x = np.array(x)

        x_std = (x + np.ones(x.shape)) / 2
        return x_std * (self.maxs - self.mins) + self.mins


class HyQActionScaler(HyQStateScaler):

    def __init__(self):

        HyQStateScaler.__init__(self)

        self.mins = np.array([-0.3, 0, -2, -0.3, 0, -2, -0.3, -1, 0, -0.3, -1, 0])
        self.maxs = np.array([0.1, 1, 0, 0.1, 1, 0, 0.1, 0, 2, 0.1, 0, 2])



if __name__ == "__main__":

    ss = HyQStateScaler()
    sa = HyQActionScaler()

    print sa.inverse_transform([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    print sa.inverse_transform([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    print sa.inverse_transform([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    print sa.transform([-0.2, 0.6, -1.7, -0.2, 0.6, -1.7, -0.2, -0.6, 1.7,-0.2, -0.6, 1.7])

    print ss.transform([0, -100000, -100000, -100000, -100000,
                        -0.3, 0, -2, -0.3, 0, -2, -0.3, -1, 0, -0.3, -1, 0,
                        -1.6, -3, -1.6, -1000, -1000, 0])
    print ss.transform([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    print ss.transform([1, 100000, 100000, 100000, 100000,
                        0.1, 1, 0, 0.1, 1, 0, 0.1, 0, 2, 0.1, 0, 2,
                        1.6, 3, 1.6, 1000, 1000, 1])