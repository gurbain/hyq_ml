import matplotlib.pyplot as plt
import numpy as np
from random import random
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import time

import processing


class FORCE(object):

    def __init__(self, regularization=0.0, elm=False,  err_window=10,
                 x_scaling=True, y_scaling=True, in_fct='lin', out_fct='lin',
                 delay_line_n=20, delay_line_step=2, train_dropout_period=50,
                 save_folder="", verbose=2, random_state=12):

        # META PARAMETERS
        self.save_folder = save_folder
        self.verbose = verbose
        self.alpha = regularization
        self.dropout_rate = 0
        self.dropout_period = train_dropout_period
        self.in_fct = in_fct
        self.out_fct = out_fct
        self.epoch = 0
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # DATA PROCESSING
        self.x_scaling = x_scaling
        self.y_scaling = y_scaling
        if self.x_scaling:
            if self.in_fct == 'tanh':
                self.x_scaler = MinMaxScaler((-1.5, 1.5))
            else:
                self.x_scaler = MinMaxScaler((-0.5, 1.5))
        if self.y_scaling:
            if self.out_fct == 'tanh':
                self.y_scaler = MinMaxScaler((-1.5, 1.5))
            else:
                self.y_scaler = MinMaxScaler((-0.5, 1.5))
        self.dl_n = delay_line_n
        self.dl_s = delay_line_step
        self.elm = elm
        self.err_window = err_window
        self.td = None
        if self.dl_n > 1:
            self.td = processing.TimeDelay(num=self.dl_n, step=self.dl_s)

        # ALGO DATA
        self.w = None
        self.p = None
        self.e = 1

    def printv(self, txt):

        if self.verbose >= 1:
            print(txt)

    def set_dropout_rate(self, rate):

        self.dropout_rate = rate

    def delay_line(self, x=None):
        
        if self.td:
            x = self.td.transform(x)
        
        return x
    
    def non_linearities(self, x=None):

        return x

    def init_algo(self, x, y):

        self.w = np.random.rand(x.shape[1], y.shape[1])
        self.p = np.identity(x.shape[1]) / self.alpha

    def neuron_fct(self, x, fct):

        if fct == "lin":
            return x

        if fct == "relu":
            return np.maximum(x, 0)

        if fct == "tanh":
            return np.tanh(x)

        return x

    def inv_neuron_fct(self, x, fct):

        if fct == "lin":
            return x

        if fct == "relu":
            return np.maximum(x, 0)

        if fct == "tanh":
            x[x < -0.999] = -0.999
            x[x > 0.999] = 0.999
            return np.arctan(x)

        return x

    def transform_ft(self, x=None, y=None):

        if y is None:
            # Scaling
            if self.x_scaling:
                x = np.mat(x)
                self.x_scaler.partial_fit(x)
                x = self.x_scaler.transform(x)

            # Processing
            x = np.mat(x)
            x = self.delay_line(x)
            x = self.non_linearities(x)
            return self.neuron_fct(x, self.in_fct)

        # Scaling
        if self.x_scaling:
            x = np.mat(x)
            self.x_scaler.partial_fit(x)
            x = self.x_scaler.transform(x)
        if self.y_scaling:
            y = np.mat(y)
            self.y_scaler.partial_fit(y)
            y = self.y_scaler.transform(y)

        # Processing (delay line can stay here to improve speed but non linearity should go before scaling)
        x = np.mat(x)
        y = np.mat(y)
        x = self.delay_line(x)
        x = self.non_linearities(x)

        # Neuronal function
        return self.neuron_fct(x, self.in_fct), self.neuron_fct(y, self.out_fct)

    def transform_out(self, y):

        # Neuronal function
        y = self.inv_neuron_fct(y, self.out_fct)

        # Scaling
        if self.y_scaling:
            y = self.y_scaler.inverse_transform(y)

        return y

    def fit_transform(self, x=None, y=None):

        if self.epoch % self.dropout_period >= self.dropout_period * (1 - self.dropout_rate):
            return self.transform(x)

        # Transform features
        x, y = self.transform_ft(x, y)

        # Algorithm init
        if self.epoch == 0:
            self.init_algo(x, y)

        # Update inverse Correlation Matrix of the states
        den = 1 + x * self.p * x.T
        num = self.p * x.T * x * self.p
        self.p = self.p - num / den

        # Compute minimal error a window
        self.e = x * self.w - y

        # Update weight matrix
        self.w = self.w - self.p * x.T * self.e

        # Update iteration
        self.epoch += 1

        # Update output
        return self.transform_out(x * self.w)

    def transform(self, x=None):

        if not self.epoch > 0:
            print("ERROR: The FORCE class has to be fitted before prediction")
            return -1

        self.epoch += 1
        x = self.transform_ft(x)
        return self.transform_out(x * self.w)


class TestFORCE(object):

    def __init__(self, n, t_max, m_in, m_out, reg=0.0001):

        self.n_length = n
        self.in_dim = m_in
        self.out_dim = m_out
        self.t_max = t_max
        self.reg = reg
        self.t = np.linspace(0, self.t_max, self.n_length)
        self.fs = self.n_length / self.t_max
        
        self.force = None
        self.x = self.create_in()
        self.y = self.create_out()
        self.y_pred = np.zeros(self.y.shape)
        
    def create_in(self):

        freq = [1, 5, 10, 20]
        x = np.zeros((self.n_length, self.in_dim))
        for i in range(self.in_dim):
            x_d = np.zeros(self.t.shape)
            for f in freq:
                a = 5 * random()
                phi = np.pi * 2 * random()
                b = np.random.normal(0, 0.5, self.t.shape)
                x_d += a * np.sin(2*np.pi*f*self.t + phi) + b
            x[:, i] = x_d

        return x

    def create_out(self):

        y = np.zeros((self.n_length, self.out_dim))
        for i in range(self.out_dim):
            y_port = 0.5 * np.sin(2*np.pi*self.t)
            f = 5
            phi = np.pi * 2 * random()
            y[:, i] = y_port + 0.1 * signal.square(2*np.pi*f * self.t + phi)

        return y

    def run(self):
        
        self.force = FORCE(regularization=self.reg)

        last_t = time.time()
        t_init = last_t
        last_i = -1
        for i in range(int(self.n_length/2)):
            self.y_pred[i, :] = self.force.fit_transform(self.x[i, :], self.y[i, :])
            if i % 500 == 0:
                t = time.time()
                tp = (t - last_t)
                if tp != 0:
                    f = (i - last_i) / tp
                    print("It: " + str(i) +
                          "\tReal Time: " + "{:.2f}".format(t - t_init) + " s" +
                          "\tController Freq: " + "{:.2f}".format(f) + " Hz")
                    last_i = i
                    last_t = t

        for i in range(int(self.n_length/2), self.n_length):
            self.y_pred[i, :] = self.force.transform(self.x[i, :])
            if i % 500 == 0:
                t = time.time()
                tp = (t - last_t)
                if tp != 0:
                    f = (i - last_i) / tp
                    print("It: " + str(i) +
                          "\tReal Time: " + "{:.2f}".format(t - t_init) + " s" +
                          "\tController Freq: " + "{:.2f}".format(f) + " Hz")
                    last_i = i
                    last_t = t

    def plot_in(self):

        plt.plot(self.t, self.x, linewidth=1.5)
        plt.show()

    def plot_out(self):

        plt.plot(self.t, self.y, linewidth=1.5)
        plt.plot(self.t, self.y_pred)
        plt.show()


if __name__ == "__main__":

    t = TestFORCE(4000, 16, 40, 1, 0.01)
    t.run()
    t.plot_out()
