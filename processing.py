import collections
import math
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
import time

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# MATPLOTLIB STYLE
plt.style.use('fivethirtyeight')
plt.rc('lines', linewidth= 1)
plt.rc('text', usetex=False)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)


class HyQStateScaler(BaseEstimator, TransformerMixin):
    """ Grosse merde, please don't have a look here, it's
        entirely hardcoded!
    """

    def __init__(self, n_in):

        self.n_in = n_in

        if n_in == 1:
            self.names = ["Forward Velocity"]
            mins = [0]
            maxs = [1]

        if n_in > 1 and n_in > 46:

            self.names = ["Roll Angle",
                          "Pitch Angle",
                          "Roll Vel",
                          "Pitch Vel",
                          "Forward Velocity"]
            mins = [math.radians(-90), math.radians(-90),
                    math.radians(-90), math.radians(-90), 0]
            maxs = [math.radians(90), math.radians(90),
                    math.radians(90), math.radians(90), 1]

            if n_in == 8 or n_in == 12:

                self.names[4] = ["X Vel"]
                mins[4] = [-1.5]
                maxs[4] = [1.5]
                self.names += ["Y Vel",
                               "Yaw Vel",
                               "Forward Velocity"]
                mins += [-1.5, math.radians(-90), 0]
                maxs += [1.5, math.radians(90), 1]

            if n_in == 9 or n_in == 12:

                self.names += ["LF Stance Status",
                               "RF Stance Status",
                               "LH Stance Status",
                               "RH Stance Status"]
                mins += [-0.2, -0.2, -0.2, -0.2]
                maxs += [1.2, 1.2, 1.2, 1.2]

        if n_in == 46:
            self.names = ["Roll Vel",
                          "Pitch Vel",
                          "Yaw Vel",
                          "X Vel",
                          "Y Vel",
                          "Z Vel",

                          "LF HAA Pos",
                          "LF HFE Pos",
                          "LF KFE Pos",
                          "RF HAA Pos",
                          "RF HFE Pos",
                          "RF KFE Pos",
                          "LH HAA Pos",
                          "LH HFE Pos",
                          "LH KFE Pos",
                          "RH HAA Pos",
                          "RH HFE Pos",
                          "RH KFE Pos",

                          "LF HAA Vel",
                          "LF HFE Vel",
                          "LF KFE Vel",
                          "RF HAA Vel",
                          "RF HFE Vel",
                          "RF KFE Vel",
                          "LH HAA Vel",
                          "LH HFE Vel",
                          "LH KFE Vel",
                          "RH HAA Vel",
                          "RH HFE Vel",
                          "RH KFE Vel",

                          "LF HAA Eff",
                          "LF HFE Eff",
                          "LF KFE Eff",
                          "RF HAA Eff",
                          "RF HFE Eff",
                          "RF KFE Eff",
                          "LH HAA Eff",
                          "LH HFE Eff",
                          "LH KFE Eff",
                          "RH HAA Eff",
                          "RH HFE Eff",
                          "RH KFE Eff",

                          "LF Stance",
                          "RF Stance",
                          "LH Stance",
                          "RH Stance"]

            mins = [-2, -2, -2, -0.5, -0.5, -0.5]
            for i in range(2):
                mins.append(math.radians(-90))
                mins.append(math.radians(-50))
                mins.append(math.radians(-140))
            for i in range(2):
                mins.append(math.radians(-90))
                mins.append(math.radians(-70))
                mins.append(math.radians(20))
            for i in range(12):
                mins.append(-10)
            for i in range(12):
                mins.append(-150)
            for i in range(4):
                mins.append(-1.2)

            maxs = [2, 2, 2, 0.5, 0.5, 0.5]
            for i in range(2):
                maxs.append(math.radians(30))
                maxs.append(math.radians(70))
                maxs.append(math.radians(-20))
            for i in range(2):
                maxs.append(math.radians(30))
                maxs.append(math.radians(50))
                maxs.append(math.radians(140))
            for i in range(12):
                maxs.append(10)
            for i in range(12):
                maxs.append(150)
            for i in range(4):
                maxs.append(1.2)

        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.x_scaled = None

    def _fit_transform(self, X):

        assert X.shape[1] == self.n_in, "Please init the HyQ Input " + \
                                        "Scaler with the right number" + \
                                        "of inputs"

        x_std = (X - self.mins) / (self.maxs - self.mins)
        self.x_scaled = x_std * 2 - np.ones(x_std.shape)

        return self

    def fit(self, X, y=None):

        self = self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):

        self = self._fit_transform(X)
        return self.x_scaled

    def transform(self, X):

        assert X.shape[1] == self.n_in, "Please init the HyQ Input " + \
                                        "Scaler with the right number" + \
                                        "of inputs"

        x_std = (X - self.mins) / (self.maxs - self.mins)
        self.x_scaled = x_std * 2 - np.ones(x_std.shape)

        return self.x_scaled

    def inverse_transform(self, X):

        x_std = (X + np.ones(X.shape)) / 2
        return x_std * (self.maxs - self.mins) + self.mins


class HyQJointScaler(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.names = ["LF HAA Pos",
                      "LF HFE Pos",
                      "LF KFE Pos",
                      "RF HAA Pos",
                      "RF HFE Pos",
                      "RF KFE Pos",
                      "LH HAA Pos",
                      "LH HFE Pos",
                      "LH KFE Pos",
                      "RH HAA Pos",
                      "RH HFE Pos",
                      "RH KFE Pos",

                      "LF HAA Vel",
                      "LF HFE Vel",
                      "LF KFE Vel",
                      "RF HAA Vel",
                      "RF HFE Vel",
                      "RF KFE Vel",
                      "LH HAA Vel",
                      "LH HFE Vel",
                      "LH KFE Vel",
                      "RH HAA Vel",
                      "RH HFE Vel",
                      "RH KFE Vel"]

        mins = []
        for i in range(2):
            mins.append(math.radians(-90))
            mins.append(math.radians(-50))
            mins.append(math.radians(-140))
        for i in range(2):
            mins.append(math.radians(-90))
            mins.append(math.radians(-70))
            mins.append(math.radians(20))
        for i in range(12):
            mins.append(-10)

        maxs = []
        for i in range(2):
            maxs.append(math.radians(30))
            maxs.append(math.radians(70))
            maxs.append(math.radians(-20))
        for i in range(2):
            maxs.append(math.radians(30))
            maxs.append(math.radians(50))
            maxs.append(math.radians(140))
        for i in range(12):
            maxs.append(10)

        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.x_scaled = None

    def _fit_transform(self, X):

        x_std = (X - self.mins) / (self.maxs - self.mins)
        self.x_scaled = x_std * 2 - np.ones(x_std.shape)

        return self

    def fit(self, X, y=None):

        self = self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):

        self = self._fit_transform(X)

        return self.x_scaled

    def transform(self, X):

        x_std = (X - self.mins) / (self.maxs - self.mins)
        self.x_scaled = x_std * 2 - np.ones(x_std.shape)

        return self.x_scaled

    def inverse_transform(self, X):

        x_std = (X + np.ones(X.shape)) / 2
        x_unscaled = x_std * (self.maxs - self.mins) + self.mins

        return x_unscaled


class TimeDelay(BaseEstimator, TransformerMixin):

    def __init__(self, num=5, step=1):

        self.num = num
        self.step = step

        self.mem = self.num * self.step
        self.init = False

        self.buff = None

    def fit(self, X, y=None):

        return self

    def fit_transform(self, X, y=None):

        y = self.transform(X)
        return y

    def create_buffer(self, dim):

        init_buffer = [[0] * dim for _ in range(self.mem)]
        self.buff = collections.deque(init_buffer, self.mem)

    def transform_it(self, x):

        self.buff.append(x)
        y = np.mat(list(self.buff))[-1::-self.step]
        y = y.reshape(1, y.size)[0]

        return y

    def transform(self, X):

        n = X.shape[0]
        dim = X.shape[1]
        y = np.zeros((n, dim * self.num))

        if not self.init:
            self.create_buffer(dim)
            self.init = True

        for i in range(n):
            x = X[i, :]
            if len(x.shape) == 1:
                x = x.tolist()
            else:
                x = x.tolist()[0]
            y[i, :] = self.transform_it(x)

        return y


class FFT(BaseEstimator, TransformerMixin):

    def __init__(self, ts=1, n_samples=12):

        self.freqs = np.fft.rfftfreq(n_samples, d=1.0/ts)

    def fit(self, X, y=None):

        return self

    def transform(self, X):
        nsamples, nfeatures = X.shape
        nfreqs = len(self.freqs)
        """Given a list of original data, return a list of feature vectors."""
        features1 = np.sin(2. * np.pi * self.freqs[None, None, :] * X[:, :,
                           None]).reshape(nsamples, nfeatures * nfreqs)
        features2 = np.cos(2. * np.pi * self.freqs[None, None, 1:] * X[:, :,
                           None]).reshape(nsamples, nfeatures * (nfreqs-1))
        features = np.concatenate([features1, features2], axis=1)
        return features


class SeasonalDecomposition(BaseEstimator, TransformerMixin):

    def __init__(self, ts=1):

        self.f = int(1.0/ts)
        self.scaler = MinMaxScaler((-1, 1))

    def fit(self, X, y=None):

        return self

    def plot(self, x, t, s, r):

        plt.subplot(411)
        plt.plot(x[:, 0], label='Original')
        plt.legend(loc='best')
        plt.subplot(412)
        plt.plot(t[:, 0], label='Trend')
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(s[:, 0],label='Seasonality')
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(r[:, 0], label='Residuals')
        plt.legend(loc='best')
        plt.show()

    def transform(self, X):

        decomposition = seasonal_decompose(X, freq=self.f)
        t = decomposition.trend
        s = decomposition.seasonal
        r = decomposition.resid

        y = np.nan_to_num(np.hstack((t, s, r)))
        y = self.scaler.transform(y)

        return y

    def fit_transform(self, X, y=None):

        decomposition = seasonal_decompose(X, freq=self.f)
        t = decomposition.trend
        s = decomposition.seasonal
        r = decomposition.resid

        y = np.nan_to_num(np.hstack((t, s, r)))
        y = self.scaler.fit_transform(y)

        return y


class GaussianNoise(BaseEstimator, TransformerMixin):

    def __init__(self, stdev=0.1):

        self.stdev = stdev

    def _fit_transform(self, X):

        noise = np.random.normal(0, self.stdev, X.shape)
        self.y = X + noise
        return self

    def fit(self, X, y=None):

        self = self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):

        self = self._fit_transform(X)
        return self.y

    def transform(self, X):

        # noise = np.random.normal(0, self.stdev, X.shape)
        self.y = X  # + noise
        return self.y


class Oscillator(BaseEstimator, TransformerMixin):

    def __init__(self, t, r, f=1.692):

        self.f = f
        self.t = t
        self.r = r
        self.i = int(self.r * self.t.shape[0])

    def fit(X, y=None):

        return self

    def fit_transform(self, X, y=None):

        y = np.sin(2 * np.pi * self.f * self.t[0:self.i]) - 0.5

        return y.reshape(-1, 1)

    def transform(self, X):

        n = X.shape[0]
        y = np.sin(2 * np.pi * self.f * self.t[self.i:self.i+n]) - 0.5

        # plt.plot(self.t[self.i:self.i+n], y)
        # plt.plot(self.t[self.i:self.i+n], X[:, 0])
        # plt.show()

        return y.reshape(-1,1)


if __name__ == "__main__":

    if len(sys.argv) > 1:

        if sys.argv[1] == "td":

            # Create an impulse array
            inputs = np.arange(40000*4).reshape(40000, 4)

            # Create the time delay object
            e = TimeDelay(num=4, step=2)

            # Run iteratively
            t_i = time.time()
            y = []
            for i in range(inputs.shape[0]):
                y.append(e.transform(inputs[i, :].reshape(1, 4))[0])
            y = np.mat(y)
            t_f1 = time.time()

            # Run in one block and compare
            e.init = False
            y2 = e.transform(inputs)
            t_f = time.time()
            print "First cells of Y iteratively: " + str(y[0:5, :])
            print "First cells of Y in block: " + str(y2[0:5, :])
            print "Running time: Iteratively: {0:.4f}".format(t_f1 - t_i) + \
                  "s \tBlock: {0:.4f}".format(t_f - t_f1) + "s"
