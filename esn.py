"""Simple Echo State Network
"""

# Copyright (C) 2015 Sylvain Chevallier <sylvain.chevallier@uvsq.fr>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# TODO: add n_readout = -1 for n_readout = n_components

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import math
import mdp
from numpy import zeros, ones, concatenate, array, tanh, vstack, arange
import numpy as np
from random import random, randint
import pickle
import scipy.linalg as la
from scipy import signal
import sys

warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=DeprecationWarning)
from keras.callbacks import ModelCheckpoint
from keras.datasets import boston_housing
from keras.optimizers import Adadelta
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input

warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt


## MATPLOTLIB STYLE
plt.style.use('fivethirtyeight')
plt.rc('lines', linewidth= 1)
plt.rc('text', usetex=False)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)

class HyQStateScaler(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.names = ["Roll Vel",
                      "Pitch Vel",
                      "Yaw Vel",
                      "X Vely",
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
                      "RH Stance",]

    def _fit_transform(self, X):

        self.mins = [-2, -2, -2, -0.5, -0.5, -0.5]
        for i in range(2):
            self.mins.append(math.radians(-90))
            self.mins.append(math.radians(-50))
            self.mins.append(math.radians(-140))
        for i in range(2):
            self.mins.append(math.radians(-90))
            self.mins.append(math.radians(-70))
            self.mins.append(math.radians(20))
        for i in range(12):
            self.mins.append(-10)
        for i in range(12):
            self.mins.append(-150)
        for i in range(4):
            self.mins.append(-1.2)
        self.mins = np.array(self.mins)

        self.maxs = [2, 2, 2, 0.5, 0.5, 0.5]
        for i in range(2):
            self.maxs.append(math.radians(30))
            self.maxs.append(math.radians(70))
            self.maxs.append(math.radians(-20))
        for i in range(2):
            self.maxs.append(math.radians(30))
            self.maxs.append(math.radians(50))
            self.maxs.append(math.radians(140))
        for i in range(12):
            self.maxs.append(10)
        for i in range(12):
            self.maxs.append(150)
        for i in range(4):
            self.maxs.append(1.2)
        self.maxs = np.array(self.maxs)

        assert self.mins.shape[0] == X.shape[1], \
                    "Data shape (" + str(X.shape[1]) + \
                    ") has not the correct value " + \
                    str(self.mins.shape[0])
        assert self.maxs.shape[0] == X.shape[1], \
                    "Data shape (" + str(X.shape[1]) + \
                    ") has not the correct value " + \
                    str(self.maxs.shape[0])

        X_std = (X - self.mins) / (self.maxs - self.mins)
        self.X_scaled = X_std * 2 - np.ones(X_std.shape)
        return self

    def fit(self, X, y=None):

        self = self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):

        self = self._fit_transform(X)
        return self.X_scaled

    def transform(self, X):

        X_std = (X - self.mins) / (self.maxs - self.mins)
        self.X_scaled = X_std * 2 - np.ones(X_std.shape)

        return self.X_scaled

    def inverse_transform(self, X):

        X_std = (X + np.ones(X_std.shape)) / 2
        return X_std * (self.maxs- self.mins) + self.mins



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

    def _fit_transform(self, X):

        self.mins = []
        for i in range(2):
            self.mins.append(math.radians(-90))
            self.mins.append(math.radians(-50))
            self.mins.append(math.radians(-140))
        for i in range(2):
            self.mins.append(math.radians(-90))
            self.mins.append(math.radians(-70))
            self.mins.append(math.radians(20))
        for i in range(12):
            self.mins.append(-10)
        self.mins = np.array(self.mins)

        self.maxs = []
        for i in range(2):
            self.maxs.append(math.radians(30))
            self.maxs.append(math.radians(70))
            self.maxs.append(math.radians(-20))
        for i in range(2):
            self.maxs.append(math.radians(30))
            self.maxs.append(math.radians(50))
            self.maxs.append(math.radians(140))
        for i in range(12):
            self.maxs.append(10)
        self.maxs = np.array(self.maxs)

        X_std = (X - self.mins) / (self.maxs - self.mins)
        self.X_scaled = X_std * 2 - np.ones(X_std.shape)

        return self

    def fit(self, X, y=None):

        self = self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):

        self = self._fit_transform(X)

        return self.X_scaled

    def transform(self, X):

        X_std = (X - self.mins) / (self.maxs - self.mins)
        self.X_scaled = X_std * 2 - np.ones(X_std.shape)

        return self.X_scaled

    def inverse_transform(self, X):

        X_std = (X + np.ones(X.shape)) / 2
        X_unscaled = X_std * (self.maxs- self.mins) + self.mins

        return X_unscaled


class TimeDelay(BaseEstimator, TransformerMixin):

    def __init__(self, num=5, step=1):

        self.num = num
        self.step = step

    def _fit_transform(self, X):

        self.tf = mdp.nodes.TimeDelayNode(self.num, gap=self.step)
        self.y = self.tf.execute(X)
        return self

    def fit(self, X, y=None):

        self = self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):

        self = self._fit_transform(X)
        return self.y

    def transform(self, X):

        self.y = self.tf.execute(X)

        return self.y


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

        noise = np.random.normal(0, self.stdev, X.shape)
        self.y = X #+ noise
        return self.y


class SimpleESN(BaseEstimator, TransformerMixin):

    def __init__(self, n_readout, n_components=500, damping=0.5,
                 weight_scaling=0.9, discard_steps=0, random_state=None):
        self.n_readout = n_readout
        self.n_components = n_components
        self.damping = damping
        self.weight_scaling = weight_scaling
        self.discard_steps = discard_steps
        self.random_state = check_random_state(random_state)
        self.input_weights_ = None
        self.readout_idx_ = None
        self.weights_ = None

    def _fit_transform(self, X):
        n_samples, n_features = X.shape
        X = check_array(X, ensure_2d=True)
        self.weights_ = self.random_state.rand(self.n_components, self.n_components)-0.5
        spectral_radius = np.max(np.abs(la.eig(self.weights_)[0]))
        self.weights_ *=  self.weight_scaling / spectral_radius
        self.input_weights_ = self.random_state.rand(self.n_components,
                                                         1+n_features)-0.5
        self.readout_idx_ = self.random_state.permutation(arange(1+n_features,
                                    1+n_features+self.n_components))[:self.n_readout]
        self.components_ = zeros(shape=(1+n_features+self.n_components,
                                        n_samples))

        curr_ = zeros(shape=(self.n_components, 1))
        U = concatenate((ones(shape=(n_samples, 1)), X), axis=1)
        for t in range(n_samples):
            u = array(U[t,:], ndmin=2).T
            curr_ = (1-self.damping)*curr_ + self.damping*tanh(
                self.input_weights_.dot(u) + self.weights_.dot(curr_))
            self.components_[:,t] = vstack((u, curr_))[:,0]
        return self

    def fit(self, X, y=None):
        """Initialize the network

        This is more compatibility step, as no learning takes place in the
        reservoir.

        Parameters
        ----------
        X : array-like shape, (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        self : returns an instance of self.
        """
        self = self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """Generate echoes from the reservoir.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            training set.

        Returns
        -------
        readout : array, shape (n_samples, n_readout)
            Reservoir activation generated by the readout neurons
        """
        self = self._fit_transform(X)
        return self.components_[self.readout_idx_, self.discard_steps:].T

    def transform(self, X):
        """Generate echoes from the reservoir

        Parameters
        ----------
        X : array-like shape, (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        readout : array, shape (n_samples, n_readout)
            Reservoir activation generated by the readout neurons
        """
        X = check_array(X, ensure_2d=True)
        n_samples, n_features = X.shape

        if self.weights_ is None:
            self.weights_ = self.random_state.rand(self.n_components,
                                                   self.n_components)-0.5
            spectral_radius = np.max(np.abs(la.eig(self.weights_)[0]))
            self.weights_ *=  self.weight_scaling / spectral_radius
        if self.input_weights_ is None:
            self.input_weights_ = self.random_state.rand(self.n_components,
                                                         1+n_features)-0.5
        if self.readout_idx_ is None:
            self.readout_idx_ = self.random_state.permutation(arange(1+n_features,
                                    1+n_features+self.n_components))[:self.n_readout]
        self.components_ = zeros(shape=(1+n_features+self.n_components,
                                        n_samples))

        curr_ = zeros(shape=(self.n_components, 1))
        U = concatenate((ones(shape=(n_samples, 1)), X), axis=1)
        for t in range(n_samples):
            u = array(U[t,:], ndmin=2).T
            curr_ = (1-self.damping)*curr_ + self.damping*tanh(
                self.input_weights_.dot(u) + self.weights_.dot(curr_))
            self.components_[:,t] = vstack((u, curr_))[:,0]

        return self.components_[self.readout_idx_, self.discard_steps:].T


class SFA():

    def __init__(self, x=None, y=None, pol_exp_deg=3, delay_buff_size=4):

        self.x = x
        self.y = y
        self.delay_buff_size = delay_buff_size
        self.pol_exp_deg = pol_exp_deg

        # Rescale input and output
        if self.y is not None:
            self.rescaled_y =(self.y - np.mean(self.y, 0)) / np.std(self.y, 0)
        if self.x is not None:
            self.rescaled_x =(self.x - np.mean(self.x, 0)) / np.std(self.x, 0)

    def set_input(self, x):

        self.x = x
        self.rescaled_x = (self.x - np.mean(self.x, 0)) / np.std(self.x, 0)

    def compute(self):

        self.x_slow = np.zeros((self.x.shape))
        # Stack x input in timeframes and compute a polynomial expansion
        for i in range(self.x.shape[1]):

            try:
                timeframes = mdp.nodes.TimeDelayNode(self.delay_buff_size)
                timeframed_x = \
                    timeframes.execute(self.x[:, i].reshape(self.x.shape[0], 1))
                cubic_expand = mdp.nodes.PolynomialExpansionNode(self.pol_exp_deg)
                cubic_expanded_x = cubic_expand(timeframed_x)

                # Create the SFA node and process the data
                sfa = mdp.nodes.SFANode(output_dim=1)
                slow = sfa.execute(cubic_expanded_x)
                x_slow = slow.flatten()

                # # Padding to get original dims
                # if self.time_frame_num > 1:
                #     x_slow = np.concatenate([[x_slow[0]], x_slow])
                # for i in range(self.x_slow.shape[0] - x_slow.shape[0]):
                #     x_slow = np.concatenate([x_slow, [x_slow[-1]]])
            except mdp.NodeException:
                #print "Node exception. Fill with zeros."
                x_slow = np.zeros((self.x.shape[0],))
            self.x_slow[:, i] = x_slow

        return self.x_slow

    def plot(self):

        n = self.x.shape[0]
        if self.y is not None:
            plt.plot(np.arange(n), self.rescaled_y, label='Normalized target')

        plt.plot(np.arange(n), self.rescaled_x, label="Normalized input")
        plt.plot(np.arange(n), self.x_slow, linewidth=0.7, label="SFA output")
        plt.xlabel('time t')
        plt.legend(loc='upper right')
        plt.show()


class ESNSFA(BaseEstimator, TransformerMixin):

    def __init__(self, n_readout=15, n_components=100, n_res=5,
                 l_min=-20, l_max=0, weight_scaling=0.9,
                 delay_buff_size=1):

        self.n_res = n_res
        self.n_readout = n_readout
        self.delay_buff_size = delay_buff_size
        self.n_out_single = self.n_readout * self.n_res
        self.n_out = self.n_out_single * self.delay_buff_size
        self.l_min = l_min
        self.l_max = l_max
        self.weight_scaling = weight_scaling
        self.n_components = n_components
        self.esn = []
        self.sfa = []
        self.dampings = np.logspace(self.l_min, self.l_max, n_res)
        for i in range(self.n_res):
            self.esn.append(SimpleESN(n_readout=self.n_readout,
                                 n_components=self.n_components,
                                 damping=self.dampings[i],
                                 weight_scaling=self.weight_scaling))
        for i in range(self.n_res):
            self.sfa.append(SFA(pol_exp_deg=1, delay_buff_size=5))

        return

    def _fit_transform(self, X):

        y = np.zeros((X.shape[0], int(self.n_out_single)))

        # Next ones are with ESN
        for i in range(self.n_res):
            x = self.esn[i].fit_transform(X)
            self.sfa[i].set_input(x)
            y_c = self.sfa[i].compute()
            # print "Filling SFA from: " + str(i*self.n_readout) + \
            #       "  to: " + str(i*self.n_readout+self.n_readout)
            y[:, i*self.n_readout:i*self.n_readout+self.n_readout] = y_c

        tf = mdp.nodes.TimeDelayNode(self.delay_buff_size)
        y = tf.execute(y)
        self.y = y
        return self

    def fit(self, X, y=None):

        self = self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):

        self = self._fit_transform(X)
        return self.y

    def transform(self, X):

        y = np.zeros((X.shape[0], int(self.n_out_single)))

        # Next ones are with ESN
        for i in range(self.n_res):
            x = self.esn[i].transform(X)
            self.sfa[i].set_input(x)
            y_c = self.sfa[i].compute()
            # print "Filling SFA from: " + str(i*self.n_readout) + \
            #       "  to: " + str(i*self.n_readout+self.n_readout)
            y[:, i*self.n_readout:i*self.n_readout+self.n_readout] = y_c

        tf = mdp.nodes.TimeDelayNode(self.delay_buff_size)
        y = tf.execute(y)
        self.y = y
        return self.y

    def plot_sfa(self, t=None):

        if self.n_res > 5:
            n_col = 2
            n_row = int(np.ceil(self.n_res/2))
        else:
            n_col = 1
            n_row = self.n_res
        fig, ax = plt.subplots(n_row, n_col)

        for i in range(self.n_res):
            if self.n_res > 5:
                indexes = (i%n_row, i/n_row)
            else:
                indexes = i
            y = self.y[:, i*self.n_readout:i*self.n_readout+self.n_readout]
            if t is not None:
                ax[indexes].plot(t, np.sum(y, axis=1)/y.shape[1])
            else:
                ax[indexes].plot(np.sum(y, axis=1)/y.shape[1])
        plt.show()


class TestESNSFA():

    def __init__(self):

        self.n_length = 20000
        self.t_max = 40
        self.t = np.linspace(0, self.t_max, self.n_length)
        self.fs = self.n_length / self.t_max
        self.in_dim = 24
        self.out_dim = 1
        self.direct_buff_delay = 3
        self.esn_buff_delay = 3

        self.val_split = 0.1
        self.test_split = 0.1

        self.batch_size = 2048
        self.epochs = 1500
        self.callbacks = []
        self.verbose = 1

        self.create_in()
        self.create_out()
        self.split_data()
        self.create_nn()

        return

    def scale(self):

        self.x_scaler = MinMaxScaler((-1, 1))
        self.y_scaler = MinMaxScaler((-1, 1))
        self.y_train = self.y_train.reshape(self.y_train.shape[0], 1)
        self.y_test = self.y_test.reshape(self.y_test.shape[0], 1)
        self.x_train = self.x_scaler.fit_transform(self.x_train)
        self.y_train = self.y_scaler.fit_transform(self.y_train)
        self.x_test = self.x_scaler.fit_transform(self.x_test)
        self.y_test = self.y_scaler.fit_transform(self.y_test)

    def create_in(self):

        freq = [1, 5, 10, 20]
        x = np.zeros((self.n_length, self.in_dim))
        for i in range(self.in_dim):
            x_d = np.zeros(self.t.shape)
            for f in freq:
                a = 5 * random()
                phi = np.pi * 2 * random()
                b = np.random.normal(0, 0.5, self.t.shape)
                x_d += a * np.sin(2*np.pi*f*self.t + phi)# + b
            x[:, i] = x_d

        self.x_scaler = MinMaxScaler((-1, 1))
        self.x = self.x_scaler.fit_transform(x)
        return x

    def create_out(self):

        y = np.zeros((self.n_length, self.out_dim))
        for i in range(self.out_dim):
            y_port = 0.5 * np.sin(2*np.pi*self.t)
            f = 5
            phi = np.pi * 2 * random()
            y[:, i] = y_port + 0.1 * signal.square(2*np.pi*f* self.t + phi)

        self.y = y
        return y

    def plot_in(self):

        print(self.x_train.shape)
        plt.plot(self.x_train)
        plt.show()

    def plot_sfa(self):

        self.preprocess()
        self.esnsfa.plot_sfa()#self.t[0:int(self.n_length*(1-self.test_split))])

    def plot_out(self):

        print(self.y_train.shape)
        plt.plot(self.y_train)
        #plt.ylim((-1.2, 1.2))
        plt.show()

    def split_data(self):

        # Divide into training and testing dataset
        x_train, x_test, y_train, y_test = train_test_split(self.x,
                                                            self.y,
                                                            shuffle=False,
                                                            test_size=self.test_split)
        # Print dimensions in a table
        lin = ["Input", "Output"]
        col = ["Training", "Validation", "Testing"]
        dat = [[(int(x_train.shape[0]*(1-self.val_split)), x_train.shape[1]),
                (int(y_train.shape[0]*(1-self.val_split)), y_train.shape[1])],
               [(int(x_train.shape[0]*self.val_split), x_train.shape[1]),
                (int(y_train.shape[0]*self.val_split), y_train.shape[1])],
               [x_test.shape, y_test.shape]]
        row_format = "{:>25}" * (len(lin) + 1)
        print(row_format.format("", *lin))
        for c, r in zip(col, dat):
            print(row_format.format(c, *r))

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def create_nn(self):

        self.esnsfa = ESNSFA(delay_buff_size=self.esn_buff_delay)

        n_in = self.esnsfa.n_out + self.in_dim * self.direct_buff_delay
        inp = Input(shape=(n_in, 1, ),
                            name='in')
        # Network
        x = Flatten()(inp)
        x = Dense(2048)(x)
        x = Activation('tanh')(x)
        x = Dense(self.out_dim)(x)
        x = Activation('tanh')(x)

        # Compile and print network
        self.nn = Model(inputs=[inp], outputs=x)
        self.nn.compile(loss='mse',
                   optimizer=Adadelta(),
                   metrics=['accuracy', 'mse', 'mae'])
        self.nn.summary()

    def preprocess(self, p_type="train"):

        if p_type == "train":
            x = self.x_train
            x_esnsfa = self.esnsfa.fit_transform(x)
            y = self.y_train
        else:
            x = self.x_test
            x_esnsfa = self.esnsfa.transform(x)
            y = self.y_test

        tf = mdp.nodes.TimeDelayNode(self.direct_buff_delay)
        x_direct = tf.execute(x)

        x = np.hstack((x_esnsfa, x_direct))
        x = np.expand_dims(x, axis=2)
        return x, y

    def fit(self):

        x, y = self.preprocess("train")

        # Train the network
        self.callbacks = [ModelCheckpoint("best.h5",
                                monitor='val_loss', verbose=self.verbose,
                                save_best_only=True, mode='min')]
        self.history = self.nn.fit(x, y,
                          validation_split=self.val_split,
                          batch_size=self.batch_size,
                          epochs=self.epochs,
                          callbacks=self.callbacks,
                          verbose=self.verbose)

    def evaluate(self, show=True):

        # Process features
        x, y = self.preprocess("test")

        # Process NN
        score = self.nn.evaluate(x, y, verbose=2)
        y_pred = self.nn.predict(x, batch_size=self.batch_size,
                                 verbose=self.verbose)

        print("Test loss: " + str(score[0]))
        print("Test accuracy: " + str(score[1]))
        self.test_loss = score[0]
        self.test_accuracy = score[1]

        if show:
            # Summarize history for accuracy
            h = self.history.history
            # plt.plot(h['acc'])
            # plt.plot(h['val_acc'])
            # plt.title('Model accuracy')
            # plt.ylabel('Accuracy')
            # plt.xlabel('Epoch')
            # plt.legend(['train', 'test'], loc='upper left')
            # plt.show()

            # Summarize history for loss
            plt.plot(h['loss'])
            plt.plot(h['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

            # Plot test and predicted values
            plt.plot(y, y_pred, marker='o', linestyle='None')
            plt.xlabel("Actual value")
            plt.ylabel("Predicted value")
            plt.show()
            plt.plot(self.t[-self.x_test.shape[0]:],
                     y[:, 0], label="real")
            plt.plot(self.t[-self.x_test.shape[0]:],
                     y_pred[:, 0], label="predicted")
            plt.plot(self.t[-self.x_test.shape[0]:],
                     np.abs(y - y_pred)[:, 0], label="MAE error")
            plt.legend()
            plt.show()

        return y, y_pred, score


if __name__ == "__main__":

    if len(sys.argv) > 1:

        if sys.argv[1] == "impulse":
            # Create an impulse array
            inputs = np.zeros((4000, 25))
            inputs[1100:1110, :] = 2.0
            inputs[2300:2350, :] = 2.0
            print(inputs)

            # Create and run reservoir
            e = SimpleESN(n_readout=1, n_components=100, damping=0.9,
                 weight_scaling=1.3)
            y = e.transform(inputs)
            plt.plot(y)
            plt.show()

        if sys.argv[1] == "lambda":
            # Create an impulse array
            inputs = np.zeros((120, 25))
            inputs[0:10, :] = 2.0
            plt.plot(inputs[:, 0])

            # Create and run reservoir
            for a in [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]:
                e = SimpleESN(n_readout=1, n_components=100, damping=a,
                              weight_scaling=0.9)
                y = e.transform(inputs)
                plt.plot(y, label=str(a))
            plt.legend()
            plt.show()

        if sys.argv[1] == "sfa":

            # Create Input and Target
            n = 300
            S = np.zeros((n, 1), 'd')
            D = np.zeros((n, 1), 'd')
            S[0] = 0.6
            for t in range(1, n):
                D[t] = np.sin(np.pi/75. * t) - t/150.
                S[t] = (3.7+0.35*D[t]) * S[t-1] * (1 - S[t-1])

            # Create SFA instance, compute and plot
            sfa = SFA(S, D)
            S_slow = sfa.compute()
            sfa.plot()


        if sys.argv[1] == "esnsfa":

            t = TestESNSFA()
            #t.plot_in()
            #t.plot_out()
            t.plot_sfa()
            t.fit()
            t.evaluate(True)