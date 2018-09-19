import copy
import numpy as np
import os
import pickle
from scipy import signal
import scipy.linalg as la
import sys
import time

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_random_state, check_array

import keras
from keras.layers import Activation, Dense, Input, Flatten
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf

import utils

import matplotlib.pyplot as plt

# MATPLOTLIB STYLE
plt.style.use('fivethirtyeight')
plt.rc('lines', linewidth=1)
plt.rc('text', usetex=False)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)


class ForwardESN(BaseEstimator, TransformerMixin):

    def __init__(self, n_readout, n_components=500, damping=0.5, n_bias=10,
                 bias_val=0.1, weight_scaling=0.9, discard_steps=0,
                 random_state=None):

        self.n_readout = n_readout
        self.n_bias = n_bias
        self.bias_val = bias_val
        self.n_components = n_components
        self.damping = damping
        self.weight_scaling = weight_scaling
        self.discard_steps = discard_steps
        self.random_state = check_random_state(random_state)
        self.input_weights_ = None
        self.readout_idx_ = None
        self.bias_ = None
        self.weights_ = None
        self.components_ = None

        # Keep the states in memory if the transform method is called several times successively
        self.curr_ = None

    def _fit_transform(self, x):

        n_samples, n_features = x.shape
        x = check_array(x, ensure_2d=True)
        self.weights_ = self.random_state.rand(self.n_components, self.n_components)-0.5
        spectral_radius = np.max(np.abs(la.eig(self.weights_)[0]))
        self.weights_ *= self.weight_scaling / spectral_radius
        self.input_weights_ = self.random_state.rand(self.n_components, 1+n_features)-0.5
        self.readout_idx_ = self.random_state.permutation(np.arange(1+n_features, 1 +
                                                                    n_features+self.n_components))[:self.n_readout]
        bias_idx_ = self.random_state.permutation(np.arange(0, self.n_components))[:self.n_bias]
        self.bias_ = np.zeros(shape=(self.n_components, 1))
        self.bias_[bias_idx_] = self.bias_val
        self.components_ = np.zeros(shape=(1+n_features+self.n_components, n_samples))

        curr_ = np.zeros(shape=(self.n_components, 1))
        u = np.concatenate((np.ones(shape=(n_samples, 1)), x), axis=1)
        for t in range(n_samples):
            u_row = np.array(u[t, :], ndmin=2).T
            curr_ = (1-self.damping)*curr_ + self.damping*np.tanh(self.input_weights_.dot(u_row) +
                                                                  self.weights_.dot(curr_)) + self.bias_
            self.components_[:, t] = np.vstack((u_row, curr_))[:, 0]
        return self

    def fit(self, x):

        self = self._fit_transform(x)
        return self

    def fit_transform(self, x, y=None, **kwargs):

        self = self._fit_transform(x)
        return self.components_[self.readout_idx_, self.discard_steps:].T

    def transform(self, x):

        x = check_array(x, ensure_2d=True)
        n_samples, n_features = x.shape

        if self.weights_ is None:
            self.weights_ = self.random_state.rand(self.n_components,
                                                   self.n_components)-0.5
            spectral_radius = np.max(np.abs(la.eig(self.weights_)[0]))
            self.weights_ *= self.weight_scaling / spectral_radius
        if self.input_weights_ is None:
            self.input_weights_ = self.random_state.rand(self.n_components,
                                                         1+n_features)-0.5
        if self.readout_idx_ is None:
            self.readout_idx_ = self.random_state.permutation(np.arange(1+n_features, 1+n_features +
                                                                        self.n_components))[:self.n_readout]
        if self.bias_ is None:
            bias_idx_ = self.random_state.permutation(np.arange(0, self.n_components))[:self.n_bias]
            self.bias_ = np.zeros(shape=(self.n_components, 1))
            self.bias_[bias_idx_] = self.bias_val

        self.components_ = np.zeros(shape=(1+n_features+self.n_components, n_samples))

        if self.curr_ is None:
            self.curr_ = np.zeros(shape=(self.n_components, 1))
        u = np.concatenate((np.ones(shape=(n_samples, 1)), x), axis=1)
        for t in range(n_samples):
            u_row = np.array(u[t, :], ndmin=2).T
            self.curr_ = (1-self.damping)*self.curr_ + \
                         self.damping*np.tanh(self.input_weights_.dot(u_row) + self.weights_.dot(self.curr_)) + \
                         self.bias_
            self.components_[:, t] = np.vstack((u_row, self.curr_))[:, 0]

        return self.components_[self.readout_idx_, self.discard_steps:].T


class FeedbackESN(BaseEstimator):

    def __init__(self, n_inputs, n_outputs, in_esn_mask, out_esn_mask,
                 n_reservoir=200, n_read=100, spectral_radius=0.95, damping=0.1, sparsity=0.0, noise=0.001,
                 fb_scaling=None, fb_shift=None, keras_model=None, real_fb=False, out_activation=utils.identity,
                 inverse_out_activation=utils.identity, random_state=None, verbose=1):
        """
        Args:
            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            in_esn_mask: a boolean mask of input size. The true elements are
                         injected in the ESN, the false, directly in the readout
            out_esn_mask: a boolean mask of output size. The true elements are fed
                          back to the ESN
            n_reservoir: nr of reservoir neurons
            n_read: nr of reservoir random neurons connected to the readout
            spectral_radius: spectral radius of the recurrent weight matrix
            damping: the leakage value during neuron activation
            sparsity: proportion of recurrent weights set to zero
            noise: noise added to each neuron (regularization)
            fb_scaling: factor applied to the signals fed back in the ESN
            fb_shift: additive term applied to the signals fed back in the ESN
            keras_model: trained to map the readout to the output.
                         input_size must be n_reservoir+1; output_size should
                         be the same as n_outputs. Call `compile` on the model
                         before passing it as a parameter
            out_activation: output activation function (applied to the readout).
                            Only valid if no keras model is specified
            inverse_out_activation: inverse of the output activation function.
                                    Only valid if no keras model is specified
            random_state: positive integer seed, np.rand.RandomState object,
                          or None to use numpy's builting RandomState.
            verbose: verbose level
        """

        # Reservoir parameters
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.in_esn_mask = in_esn_mask
        self.out_esn_mask = out_esn_mask
        self.n_in_esn = sum(self.in_esn_mask)
        self.n_out_esn = sum(self.out_esn_mask)
        self.skip_esn_id = [not i for i in self.in_esn_mask]
        self.n_reservoir = n_reservoir
        self.n_read = n_read
        self.spectral_radius = spectral_radius
        self.damping = damping
        self.sparsity = sparsity
        self.noise = noise
        self.real_fb = real_fb

        # Readout parameters
        self.fb_scaling = fb_scaling
        self.fb_shift = fb_shift
        self.keras_model = keras_model
        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation

        # Miscellaneous parameters
        self.random_state = random_state
        self.verbose = verbose

        # States
        self.W = None
        self.W_fb = None
        self.W_in = None
        self.W_out = None
        self.rd_id = None
        self.states = None
        self.state_it = 0
        self.laststate = None
        self.lastinput = None
        self.lastoutput = None

        # Create the random state
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as err:
                raise Exception("Invalid seed: " + str(err))
        else:
            self.random_state_ = np.random.mtrand._rand

        # Init the reservoir
        self.init_weights()

    def init_weights(self):
        """
        Initialize the states of the network
        """

        # Initialize recurrent weights:
        # Begin with a random matrix centered around zero:
        w = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # Delete the fraction of connections given by (self.sparsity):
        w[self.random_state_.rand(*w.shape) < self.sparsity] = 0
        # Compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eigvals(w)))
        # Rescale them to reach the requested spectral radius:
        self.W = w * (self.spectral_radius / radius)

        # Random input weights:
        self.W_in = self.random_state_.rand(self.n_reservoir, self.n_in_esn) * 2 - 1

        # Random feedback (teacher forcing) weights:
        self.W_fb = self.random_state_.rand(self.n_reservoir, self.n_out_esn) * 2 - 1

        # Random selection of reservoir neurons to inject in the readout
        self.rd_id = self.random_state_.permutation(np.arange(0, self.n_reservoir))[:self.n_read]

    def _scale_out(self, out):
        """
        Multiplies the feedback signal by the bd_scaling argument,
        then adds the teacher_shift argument to it.
        """

        if self.fb_scaling is not None:
            out = out * self.fb_scaling
        if self.fb_shift is not None:
            out = out + self.fb_shift

        return out

    def _unscale_out(self, out_scaled):
        """
        Inverse operation of the _scale_fb method.
        """

        if self.fb_shift is not None:
            out_scaled = out_scaled - self.fb_shift
        if self.fb_scaling is not None:
            out_scaled = out_scaled / self.fb_scaling

        return out_scaled

    def _update(self, state, input_pattern, output_pattern):
        """
        Performs one update step. I.e., computes the next network state by applying
        the recurrent weights to the last state & and feeding in the current input
        and output patterns.
        """
        # print "computing states: " + str(self.state_it)

        self.state_it += 1
        preactivation = np.dot(self.W, state) + \
                        np.dot(self.W_in, input_pattern) + \
                        np.dot(self.W_fb, output_pattern)

        arg = (np.tanh(preactivation) # ADD DAMPING
                + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5))

        return arg

    def fit(self, x, y, inspect=False, state_update=True, win=1000, **kwargs):
        """
        Collect the network's reaction to training data, train readout weights.

        Args:
            x: np.array of dimensions (N_training_samples x n_inputs)
            y: np.array of dimension (N_training_samples x n_outputs)
            inspect: show a visualisation of the collected reservoir states

        Returns:
            the network's output on the training data, using the trained weights
        """

        # PRE-PROCESSING
        # Transform any vectors of shape (x,) into vectors of shape (x,1):
        if x.ndim < 2:
            x = np.reshape(x, (len(x), -1))
        if y.ndim < 2:
            y = np.reshape(y, (len(y), -1))
        print x.shape, y.shape
        # Transform input and feedback signal:
        y_esn = self._scale_out(y)[:, self.out_esn_mask]
        x_esn = x[:, self.in_esn_mask]
        n_samples = x_esn.shape[0]

        if not self.real_fb or self.keras_model is None:
            # STATES UPDATE
            if state_update:
                if self.verbose > 0:
                    print("\n\n ===== Harvesting States =====\n")
                self.states_t = np.zeros((n_samples, self.n_reservoir))
                for n in range(1, n_samples):
                    self.states_t[n, :] = self._update(self.states_t[n - 1], 
                                                       np.ravel(x_esn[n, :]), np.ravel(y_esn[n - 1, :]))

            # READOUT LEARNING
            if self.verbose > 0:
                print("\n\n ===== Fitting Readout =====\n")
            # Disregard the first transient states
            transient = min(int(x_esn.shape[1] / 10), 100)
            # Include the inputs that shall skip the ESN
            states_ext = np.hstack((self.states_t[:, self.rd_id], x[:, self.skip_esn_id]))
            # Use scipy linear regression or keras back-propagation
            if self.keras_model is None:
                # Solve for W_out:
                self.W_out = np.dot(np.linalg.pinv(states_ext[transient:, :]),
                                    self.inverse_out_activation(y[transient:, :])).T
                # Apply learned weights to the collected states:
                pred_train = self._unscale_out(self.out_activation(np.dot(states_ext, self.W_out.T)))
                pred_train[1:] = self.out_activation(pred_train[1:])
            else:
                states_ext_exp = np.expand_dims(states_ext, axis=2)
                # Train
                self.keras_model.fit(states_ext_exp[transient:, :], y[transient:, :], **kwargs)
                # Predict
                pred_train = self._unscale_out(self.keras_model.predict(states_ext_exp))

        # Learn the readout with every sample sequentially and use the real feedback
        else:
            if self.verbose > 0:
                print("\n\n ===== Updating ESN + Readout Sequentially =====\n")

            for i in range(1):
                self.states_t = np.zeros((n_samples, self.n_reservoir))
                pred_train = copy.copy(y)
                # mult = float(i) / 50
                batch_size = 2000
                for n in range(1, n_samples):
                    # Predict NN fb and mix with real
                    states_ext_exp_n = np.expand_dims(np.hstack([self.states_t[n-1, self.rd_id].reshape(1, -1),
                                                                      x[n, self.skip_esn_id].reshape(1, -1)]), axis=2)
                    pred_train[n-1, :] = self.keras_model.predict_on_batch(states_ext_exp_n)
                    mult = min(float(n) / (n_samples * 2), 1)
                    y_mix = mult * self._scale_out(pred_train[n-1, self.out_esn_mask]) + (1-mult) * y_esn[n-1, :]
                    y_noise = y_mix + 0.01 * np.random.normal(size=y_esn[n-1, :].shape)

                    # Update ESN
                    self.states_t[n, :] = self._update(self.states_t[n-1, :],
                                                       np.ravel(x_esn[n, :]), np.ravel(y_noise))

                    # When end of batch size:
                    if n % batch_size == 0 and n > (batch_size - 1):

                        if n % batch_size == 0:
                            # callbacks
                            pass

                        # Train on batch
                        states_ext = np.hstack([self.states_t[n+1-batch_size:n+1, self.rd_id],
                                                     x[n+1-batch_size:n+1, self.skip_esn_id]])
                        #states_ext += 0.01 * np.random.normal(size=states_ext.shape)
                        states_ext_exp = np.expand_dims(states_ext.reshape(batch_size, -1), axis=2)
                        for _ in range(400):
                            h = self.keras_model.train_on_batch(states_ext_exp,
                                                                y_esn[n+1-batch_size:n+1, :].reshape(batch_size, -1))

                        if n % batch_size == 0:
                            # callbacks
                            pass

                    if n % 6000 == 0:
                        print mult, h
                        plt.plot(y[0:n, 1])
                        plt.plot(pred_train[0:n, 1])
                        plt.show()

        # CONTINUATION:
        # Remember the last state for later
        self.laststate = self.states_t[-1, :]
        self.lastinput = x[-1, :]
        self.lastoutput = y[-1, :]

        # VIZUALIZATION
        if inspect:
            plt.figure(figsize=(win * 0.0025, self.states_t.shape[1] * 0.01))
            plt.imshow(states_ext.T[-win:, :], aspect='auto', interpolation='nearest')
            plt.colorbar()
            plt.show()
        if self.verbose > 0:
            print("\nTraining error: " + str(np.sqrt(np.mean((pred_train - y)**2))))

        return pred_train

    def predict(self, x, continuation=True, inspect=False, win=1000, **kwargs):
        """
        Apply the learned weights to the network's reactions to new input.

        Args:
            x: np.array of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training state
            inspect: show a visualisation of the collected reservoir states

        Returns:
            np.array of output activations
        """

        # PRE-PROCESSING
        if x.ndim < 2:
            x = np.reshape(x, (len(x), -1))
        n_samples = x.shape[0]
        # Remember the last state for later
        if continuation and self.lastinput is not None and self.laststate is not None and \
            self.lastoutput is not None:
                laststate = self.laststate
                lastinput = self.lastinput
                lastoutput = self.lastoutput
        else:
            laststate = np.zeros(self.n_reservoir)
            lastinput = np.zeros(self.n_inputs)
            lastoutput = np.zeros(self.n_outputs)
        x = np.vstack([lastinput, x])
        y = np.vstack([lastoutput, np.zeros((n_samples, self.n_outputs))])
        self.states_p = np.vstack([laststate, np.zeros((n_samples, self.n_reservoir))])
        y_esn = y[:, self.out_esn_mask]
        x_esn = x[:, self.in_esn_mask]

        # PREDICTION
        if self.verbose > 0:
            print("\n\n ===== Predicting Reservoir + Readout =====\n")
        for n in range(n_samples):
            self.states_p[n + 1, :] = self._update(self.states_p[n, :],
                                                   np.ravel(x_esn[n + 1, :]), np.ravel(y_esn[n, :]))
            states_ext = np.hstack([self.states_p[n + 1, self.rd_id].reshape(1, -1),
                                    x[n + 1, self.skip_esn_id].reshape(1, -1)])

            if self.keras_model is None:
                y[n + 1, :] = self.out_activation(np.dot(self.W_out, states_ext))
            else:
                states_ext_exp = np.expand_dims(states_ext.reshape(1, -1), axis=2)
                y[n + 1, :] = self.keras_model.predict(states_ext_exp, **kwargs)
            y_esn[n + 1, :] = y[n + 1, self.out_esn_mask]
        if self.keras_model is None:
            y[1:] = self.out_activation(y[1:])

        # VIZUALIZATION
        if inspect:
            plt.figure(figsize=(win * 0.0025, self.states_p.shape[1] * 0.01))
            plt.imshow(self.states_p.T[:win, :], aspect='auto', interpolation='nearest')
            plt.colorbar()
            plt.show()

        return self._unscale_out(y[1:])

    def evaluate(self, x, y, continuation=True, inspect=False, win=1000, **kwargs):

        y_pred = self.predict(x, continuation, inspect, win, **kwargs)
        return [np.sqrt(np.mean((y - y_pred) ** 2))]

    def save(self, folder, index=0):

        if self.verbose > 2:
            print("\n\n ===== Saving =====\n")

        # Create the folder
        utils.make_keras_picklable()
        utils.mkdir(folder)

        # Save training data
        to_save = copy.copy(self.__dict__)
        del to_save["keras_model"]
        pickle.dump(to_save, open(folder + "/esn_" + str(index) + ".pkl", "wb"), protocol=2)
        del to_save

        # Save nn model
        if self.keras_model is not None:
            if hasattr(self.keras_model, "model"):
                self.keras_model.model.save(folder+ "/readout_" + str(index) + ".h5")
            else:
                self.keras_model.save(folder+ "/readout_" + str(index) + ".h5")

    def load(self, folder, index=0):

        # Load this class
        verbose_rem = self.verbose
        with open(folder + "/esn_" + str(index) + ".pkl", 'rb') as f:

            if sys.version_info[0] >= 3:
                self.__dict__ = pickle.load(f, encoding="latin1")
            else:
                self.__dict__ = pickle.load(f)

            self.verbose = verbose_rem

        # Load network
        readout_name = folder + "/readout_" + str(index) + ".h5"
        if os.path.isfile(readout_name):
            self.keras_model = load_model(readout_name)


class TestFeedbackESN(object):

    def __init__(self):

        self.t = None
        self.inputs_train = None
        self.inputs_test = None
        self.outputs_train = None
        self.outputs_test = None
        self.in_esn_mask = None
        self.out_esn_mask = None

        self.n_in = 3
        self.n_out = 2
        self.n_read = 250
        self.n_res = 300

        self.esn = None
        self.readout = None

        self.network_layers = [[('relu', 40)], [('relu', 40)], [('relu', 40)]]
        self.val_split = 0.1
        self.batch_size = 1000
        self.max_epochs = 50
        self.regularization = 0.001
        self.metric = "mae"
        self.optim = "adam"
        self.callbacks = []
        self.verbose = 1
        self.random_state = 42
        np.random.seed(self.random_state)

    def create_in(self):

        self.t = np.linspace(0, 20, 10000)
        inputs_esn = np.zeros((self.t.shape[0], 2))
        inputs_readout = np.zeros((self.t.shape[0], 1))
        inputs_esn[:, 0] = signal.sawtooth(2 * np.pi * self.t)
        inputs_esn[:2300, 1] = -0.4
        inputs_esn[2300:8000, 1] = 0.6
        inputs_esn[8000:, 1] = -0.4
        inputs_readout[:2300, 0] = 0.2
        inputs_readout[2300:8000, 0] = 0.5
        inputs_readout[8000:, 0] = 0.2
        self.inputs_train = np.hstack((inputs_esn[:6000], inputs_readout[:6000]))
        self.inputs_test = np.hstack((inputs_esn[6000:], inputs_readout[6000:]))
        self.in_esn_mask = [True, True, False]

    def create_out(self):

        outputs = np.zeros((self.t.shape[0], 2))
        outputs[:, 0] = 0.3 * signal.square(2 * np.pi * self.t)
        outputs[:2300, 1] = 0.3 * np.sin(2 * np.pi * self.t[:2300])
        outputs[2300:8000, 1] = 0.8 * np.cos(2 * np.pi * self.t[2300:8000])
        outputs[8000:, 1] = 0.3 * np.sin(2 * np.pi * self.t[8000:])
        self.outputs_train = outputs[:6000, :]
        self.outputs_test = outputs[6000:, :]
        self.out_esn_mask = [True, False]

    def plot_in_out(self):

        plt.plot(self.t, self.inputs_train)
        plt.plot(self.t, self.outputs_train)
        plt.show()

    def create_readout_fn(self):

        # Input Layer
        n_in = self.n_read + sum([not i for i in self.in_esn_mask])
        state_input = Input(shape=(n_in, 1,),
                            name='robot_state')
        x = Flatten()(state_input)

        # Network layers
        for l in self.network_layers:
            if l[0][0] in ['relu', 'tanh']:
                reg = l2(self.regularization)
                x = Dense(l[0][1], kernel_regularizer=reg)(x)
                x = Activation(l[0][0])(x)

            # LSTM network whould be first
            if l[0][0] == "lstm":
                x = LSTM(l[0][1], activation='relu')(state_input)

        # Output Layer
        n_out = self.n_out
        x = Dense(n_out, kernel_regularizer=l2(self.regularization))(x)

        # Compile and print network
        nn_fn = Model(inputs=[state_input], outputs=x)
        keras.losses.custom_loss = self.metric
        nn_fn.compile(loss=keras.losses.custom_loss,
                      optimizer=self.optim,
                      metrics=['accuracy', 'mse', 'mae', 'mape', 'cosine'])
        if self.verbose > 1 or self.verbose == -1:
            nn_fn.summary()

        return nn_fn

    def create_nn(self):

        # Create the Keras model
        if self.max_epochs != 0:
            self.readout = KerasRegressor(build_fn=self.create_readout_fn,
                                          validation_split=self.val_split,
                                          batch_size=self.batch_size,
                                          epochs=self.max_epochs,
                                          callbacks=self.callbacks,
                                          verbose=self.verbose)
        else:
            self.readout = None

        # Create a random reservoir with bias
        self.esn = FeedbackESN(n_inputs=self.n_in, n_outputs=self.n_out, n_read=self.n_read,
                               in_esn_mask=self.in_esn_mask, out_esn_mask=self.out_esn_mask,
                               n_reservoir=self.n_res, spectral_radius=1.5, damping=0.1, sparsity=0.5, noise=0.001,
                               fb_scaling=1, fb_shift=0, keras_model=self.readout,
                               random_state=self.random_state, verbose=self.verbose)

    def fit_nn(self):

        pred_out = self.esn.fit(self.inputs_train, self.outputs_train, inspect=False)
        # plt.plot(self.t[:6000], self.outputs_train, label="Target")
        # plt.plot(self.t[:6000], pred_out, label="Prediction")
        # plt.legend()
        # plt.show()

    def evaluate(self):

        pred_out = self.esn.predict(self.inputs_test, continuation=True, inspect=False)
        if self.verbose > 0:
            print("\nTest error: " + str(np.sqrt(np.mean((pred_out - self.outputs_test) ** 2))))
        plt.plot(self.t[6000:], self.outputs_test, label="Target")
        plt.plot(self.t[6000:], pred_out, label="Prediction")
        plt.legend()
        plt.show()


if __name__ == "__main__":

    if len(sys.argv) > 1:

        if sys.argv[1] == "impulse":
            # Create an impulse np.array
            inputs = np.zeros((4000, 25))
            inputs[1100:1110, :] = 2.0
            inputs[2300:2350, :] = 2.0

            # Create a random reservoir
            e = ForwardESN(n_readout=1, n_components=100, damping=0.05,
                           weight_scaling=0.98)

            # Run iteratively
            t_i = time.time()
            y = []
            for i in range(inputs.shape[0]):
                y.append(e.transform(inputs[i, :].reshape(1, 25))[0, 0])
            t_f1 = time.time()

            # Run in one block and compare
            e.curr_ = None
            y2 = e.transform(inputs)
            t_f = time.time()
            print("Running time: Iteratively: {0:.4f}".format(t_f1 - t_i) + \
                  "s \tBlock: {0:.4f}".format(t_f - t_f1) + "s")
            plt.plot(y)
            plt.plot(y2)
            plt.show()

        if sys.argv[1] == "lambda":
            # Create an impulse np.array
            inputs = np.zeros((120, 25))
            inputs[0:10, :] = 2.0
            plt.plot(inputs[:, 0])

            # Create and run reservoir
            for a in [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]:
                e = ForwardESN(n_readout=1, n_components=100, damping=a,
                               weight_scaling=0.9)
                y = e.transform(inputs)
                plt.plot(y, label=str(a))
            plt.legend()
            plt.show()

        if sys.argv[1] == "bias":

            # Create an impulse np.array
            inputs = np.zeros((4000, 25))

            # Create a random reservoir with bias
            e = ForwardESN(n_readout=1, n_components=100, damping=0.05,
                           weight_scaling=0.98, bias_val=0.1, n_bias=10)

            # Run and show
            y = e.transform(inputs)
            plt.plot(y)
            plt.show()

        if sys.argv[1] == "esnfb":

            t = TestFeedbackESN()
            t.create_in()
            t.create_out()
            # t.plot_in_out()
            t.create_nn()
            t.fit_nn()
            t.esn.save("a")
            t.evaluate()

            # Delete, reload and test
            del t
            t = TestFeedbackESN()
            t.create_in()
            t.create_out()
            t.create_nn()
            t.esn.load("a")
            t.evaluate()
