import collections
import math
import numpy as np
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
import time

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# MATPLOTLIB STYLE
plt.style.use('fivethirtyeight')
plt.rc('lines', linewidth=1)
plt.rc('text', usetex=False)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)

class ELM(BaseEstimator, TransformerMixin):

    def __init__(self, n_in, n_elm=20, fct="tanh", scaling=True):

        self.n_in = n_in
        self.n_elm = n_elm
        self.fct = fct
        self.scaling = scaling

        if self.scaling:
            self.scaler = MinMaxScaler((-1.5, 1.5))

        self.w = np.random.rand(self.n_elm, self.n_in) - 0.5

    def fit(self):

        return self

    def fit_transform(self, x, y=None, **kwargs):

        if self.scaling:
            x = np.mat(x)
            self.scaler.partial_fit(x)

        y = self.transform(x)
        return y

    def transform(self, x):

        if self.scaling:
            x = self.scaler.transform(x)
            x = np.mat(x)

        y = (self.w * x.T).T

        if self.fct == "tanh":
            return np.mat(np.tanh(y))

        if self.fct == "relu":
            return np.maximum(y, 0)

        plt.plot(y)
        plt.show()
        return np.mat(y)


class TimeDelay(BaseEstimator, TransformerMixin):

    def __init__(self, num=5, step=1):

        self.num = num
        self.step = step

        self.mem = self.num * self.step
        self.init = False

        self.buff = None

    def fit(self):

        return self

    def fit_transform(self, x, y=None, **kwargs):

        y = self.transform(x)
        return y

    def create_buffer(self, dim):

        init_buffer = [[0] * dim for _ in range(self.mem)]
        self.buff = collections.deque(init_buffer, self.mem)

    def transform_it(self, x):

        self.buff.append(x)
        y = np.mat(list(self.buff))[-1::-self.step]
        y = y.reshape(1, y.size)[0]

        return y

    def transform(self, x):

        n = x.shape[0]
        dim = x.shape[1]
        y = np.zeros((n, dim * self.num))

        if not self.init:
            self.create_buffer(dim)
            self.init = True

        for i in range(n):
            x_row = x[i, :]
            if len(x.shape) == 1:
                x_row = x_row.tolist()
            else:
                x_row = x_row.tolist()[0]
            y[i, :] = self.transform_it(x_row)

        return np.mat(y)


class LowPassFilter(BaseEstimator, TransformerMixin):

    def __init__(self, fc=80, ts=0.004, ord=5):

        self.fc = fc                 # Hz
        self.ord = ord               # Filter order
        self.ts = ts

        self.nyq = 0.5 / self.ts
        self.norm_fc = self.fc / self.nyq

        self.filt_b = signal.firwin(self.ord, self.norm_fc)
        self.filt_a = [1]
        self.zi = None

    def fit(self):

        return self

    def fit_transform(self, x, y=None, **kwargs):

        y = self.transform(x)
        return y

    def transform(self, x):

        n = x.shape[0]
        dim = x.shape[1]
        y = np.zeros((n, dim))

        if self.zi is None:
            self.zi = []
            for i in range(dim):
                self.zi.append(signal.lfilter_zi(self.filt_b, self.filt_a) * 0.0)

        for i in range(n):
            for j in range(dim):
                y[i, j], self.zi[j] = signal.lfilter(self.filt_b, self.filt_a,
                                                     np.array([x[i, j]]), zi=np.array(self.zi[j]))

        return np.mat(y)


class FFT(BaseEstimator, TransformerMixin):

    def __init__(self, ts=1, n_samples=12):

        self.freqs = np.fft.rfftfreq(n_samples, d=1.0/ts)

    def fit(self):

        return self

    def transform(self, x):
        nsamples, nfeatures = x.shape
        nfreqs = len(self.freqs)
        """Given a list of original data, return a list of feature vectors."""
        features1 = np.sin(2. * np.pi * self.freqs[None, None, :] * x[:, :,
                           None]).reshape(nsamples, nfeatures * nfreqs)
        features2 = np.cos(2. * np.pi * self.freqs[None, None, 1:] * x[:, :,
                           None]).reshape(nsamples, nfeatures * (nfreqs-1))
        features = np.concatenate([features1, features2], axis=1)
        return features


class SeasonalDecomposition(BaseEstimator, TransformerMixin):

    def __init__(self, ts=1):

        self.f = int(1.0/ts)
        self.scaler = MinMaxScaler((-1, 1))

    def fit(self):

        return self

    def plot(self, x, t, s, r):

        plt.subplot(411)
        plt.plot(x[:, 0], label='Original')
        plt.legend(loc='best')
        plt.subplot(412)
        plt.plot(t[:, 0], label='Trend')
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(s[:, 0], label='Seasonality')
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(r[:, 0], label='Residuals')
        plt.legend(loc='best')
        plt.show()

    def transform(self, x):

        decomposition = seasonal_decompose(x, freq=self.f)
        t = decomposition.trend
        s = decomposition.seasonal
        r = decomposition.resid

        y = np.nan_to_num(np.hstack((t, s, r)))
        y = self.scaler.transform(y)

        return y

    def fit_transform(self, x, y=None, **kwargs):

        decomposition = seasonal_decompose(x, freq=self.f)
        t = decomposition.trend
        s = decomposition.seasonal
        r = decomposition.resid

        y = np.nan_to_num(np.hstack((t, s, r)))
        y = self.scaler.fit_transform(y)

        return y


class GaussianNoise(BaseEstimator, TransformerMixin):

    def __init__(self, stdev=0.1):

        self.stdev = stdev
        self.y = None

    def _fit_transform(self, x):

        noise = np.random.normal(0, self.stdev, x.shape)
        self.y = x + noise
        return self

    def fit(self, x):

        self = self._fit_transform(x)
        return self

    def fit_transform(self, x, y=None, **kwargs):

        self = self._fit_transform(x)
        return self.y

    def transform(self, x):

        # noise = np.random.normal(0, self.stdev, x.shape)
        self.y = x  # + noise
        return self.y


class Oscillator(BaseEstimator, TransformerMixin):

    def __init__(self, t, r, f=1.6915):

        self.f = f
        self.t = t
        self.r = r
        self.i = int(self.r * self.t.shape[0])

    def fit(self):

        return self

    def fit_transform(self, x, y=None, **kwargs):

        y = 0.5 * np.sin(2 * np.pi * self.f * self.t[0:self.i])
        
        return np.hstack((x, y.reshape(-1, 1)))

    def transform(self, x):

        n = x.shape[0]
        y = 0.5 * np.sin(2 * np.pi * self.f * self.t[self.i:self.i+n])
        # plt.plot(self.t[self.i:self.i+n], y)
        # plt.plot(self.t[self.i:self.i+n], x[:, 0])
        # plt.show()

        return np.hstack((x, y.reshape(-1, 1)))


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
            print("First cells of Y iteratively: " + str(y[0:5, :]))
            print("First cells of Y in block: " + str(y2[0:5, :]))
            print("Running time: Iteratively: {0:.4f}".format(t_f1 - t_i) + \
                  "s \tBlock: {0:.4f}".format(t_f - t_f1) + "s")

        if sys.argv[1] == "lpf":

            # Create a noisy input
            t = np.arange(0, 4, 0.004).reshape(1000, 1)
            x1 = np.sin(2 * np.pi * 1 * t)
            x2 = 0.2 * np.cos(2 * np.pi * 50 * t)
            x3 = np.random.normal(0.0, 0.1, 1000).reshape(1000, 1)
            x = x1 + x2 # + x3

            # SKLearn DUT
            print t[0:10]
            e = LowPassFilter(fc=40, ts=0.004, ord=10)
            y = e.transform(x)

            # Plot results
            plt.plot(t, x)
            plt.plot(t, y)
            plt.show()