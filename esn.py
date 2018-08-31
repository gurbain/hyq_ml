import numpy as np
import scipy.linalg as la
import sys
import time

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_random_state, check_array

import utils

import matplotlib.pyplot as plt

# MATPLOTLIB STYLE
plt.style.use('fivethirtyeight')
plt.rc('lines', linewidth= 1)
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

    def _fit_transform(self, X):

        n_samples, n_features = X.shape
        X = check_array(X, ensure_2d=True)
        self.weights_ = self.random_state.rand(self.n_components, self.n_components)-0.5
        spectral_radius = np.max(np.abs(la.eig(self.weights_)[0]))
        self.weights_ *= self.weight_scaling / spectral_radius
        self.input_weights_ = self.random_state.rand(self.n_components,
                                                         1+n_features)-0.5
        self.readout_idx_ = self.random_state.permutation(np.arange(1+n_features, 1+n_features+self.n_components))[:self.n_readout]
        bias_idx_ = self.random_state.permutation(np.arange(0,self.n_components))[:self.n_bias]
        self.bias_ = np.zeros(shape=(self.n_components, 1))
        self.bias_[bias_idx_] = self.bias_val
        self.components_ = np.zeros(shape=(1+n_features+self.n_components,
                                        n_samples))

        curr_ = np.zeros(shape=(self.n_components, 1))
        U = np.concatenate((np.ones(shape=(n_samples, 1)), X), axis=1)
        for t in range(n_samples):
            u = np.array(U[t, :], ndmin=2).T
            curr_ = (1-self.damping)*curr_ + self.damping*np.tanh(self.input_weights_.dot(u) +
                    self.weights_.dot(curr_)) + self.bias_
            self.components_[:, t] = np.vstack((u, curr_))[:, 0]
        return self

    def fit(self, X, y=None):

        self = self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):

        self = self._fit_transform(X)
        return self.components_[self.readout_idx_, self.discard_steps:].T

    def transform(self, X):

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
            self.readout_idx_ = self.random_state.permutation(np.arange(1+n_features, 1+n_features+self.n_components))[:self.n_readout]
        if self.bias_ is None:
            bias_idx_ = self.random_state.permutation(np.arange(0, self.n_components))[:self.n_bias]
            self.bias_ = np.zeros(shape=(self.n_components, 1))
            self.bias_[bias_idx_] = self.bias_val

        self.components_ = np.zeros(shape=(1+n_features+self.n_components, n_samples))

        if self.curr_ is None:
            self.curr_ = np.zeros(shape=(self.n_components, 1))
        U = np.concatenate((np.ones(shape=(n_samples, 1)), X), axis=1)
        for t in range(n_samples):
            u = np.array(U[t,:], ndmin=2).T
            self.curr_ = (1-self.damping)*self.curr_ + \
                         self.damping*np.tanh(self.input_weights_.dot(u) + self.weights_.dot(self.curr_)) + \
                         self.bias_
            self.components_[:,t] = np.vstack((u, self.curr_))[:,0]

        return self.components_[self.readout_idx_, self.discard_steps:].T


class FeedbackESN(BaseEstimator):

    def __init__(self, n_inputs, n_outputs, n_reservoir=200,
                 spectral_radius=0.95, sparsity=0, noise=0.001,
                 teacher_forcing=True, feedback_scaling=None,
                 teacher_scaling=None, teacher_shift=None, damping=0.5,
                 keras_model=None, out_activation=utils.identity,
                 inverse_out_activation=utils.identity, random_state=None,
                 silent=True):
        """
        Args:
            n_inputs: nr of input dimensions
            n_outputs: nr of output dimensions
            n_reservoir: nr of reservoir neurons
            spectral_radius: spectral radius of the recurrent weight matrix
            sparsity: proportion of recurrent weights set to zero
            noise: noise added to each neuron (regularization)
            teacher_forcing: if True, feed the target back into output units
            teacher_scaling: factor applied to the target signal
            teacher_shift: additive term applied to the target signal
            damping: damping on the neuron activation
            keras_model: trained to map the readout to the output.
                         input_size must be n_reservoir+1; output_size should
                         be the same as n_outputs. Call `compile` on the model
                         before passing it as a parameter
            out_activation: output activation function (applied to the readout)
            inverse_out_activation: inverse of the output activation function
            random_state: positive integer seed, np.rand.RandomState object,
                          or None to use numpy's builting RandomState.
            silent: supress messages
        """
        # check for proper dimensionality of all arguments and write them down.
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.damping = damping

        self.teacher_scaling = teacher_scaling
        self.teacher_shift = teacher_shift

        self.keras_model = keras_model
        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation
        self.random_state = random_state

        self.W = None
        self.W_fb = None
        self.W_in = None
        self.W_out = None

        self.laststate = None
        self.lastinput = None
        self.lastoutput = None
        self.states = None
        self.extended_states = None
        self.extended_states_2 = None

        self.state_it = 0

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as err:
                raise Exception("Invalid seed: " + str(err))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.teacher_forcing = teacher_forcing
        self.silent = silent
        self.initweights()

    def initweights(self):

        # initialize recurrent weights:
        # begin with a random matrix centered around zero:
        w = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        
        # delete the fraction of connections given by (self.sparsity):
        w[self.random_state_.rand(*w.shape) < self.sparsity] = 0
        
        # compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eigvals(w)))
        
        # rescale them to reach the requested spectral radius:
        self.W = w * (self.spectral_radius / radius)

        # random input weights:
        self.W_in = self.random_state_.rand(
            self.n_reservoir, self.n_inputs) * 2 - 1
        # random feedback (teacher forcing) weights:
        self.W_fb = self.random_state_.rand(
            self.n_reservoir, self.n_outputs) * 2 - 1

    def _update(self, state, input_pattern, output_pattern):
        """performs one update step.

        i.e., computes the next network state by applying the recurrent weights
        to the last state & and feeding in the current input and output patterns
        """
        # print "computing states: " + str(self.state_it)

        self.state_it += 1
        if self.teacher_forcing:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern)
                             + np.dot(self.W_fb, output_pattern))
        else:
            preactivation = (np.dot(self.W, state)
                             + np.dot(self.W_in, input_pattern))
        return (np.np.tanh(preactivation)
                + self.noise * (self.random_state_.rand(self.n_reservoir) - 0.5))

    def _scale_teacher(self, teacher):
        """multiplies the teacher/target signal by the teacher_scaling argument,
        then adds the teacher_shift argument to it."""

        if self.teacher_scaling is not None:
            teacher = teacher * self.teacher_scaling
        if self.teacher_shift is not None:
            teacher = teacher + self.teacher_shift
        return teacher

    def _unscale_teacher(self, teacher_scaled):
        """inverse operation of the _scale_teacher method."""

        if self.teacher_shift is not None:
            teacher_scaled = teacher_scaled - self.teacher_shift
        if self.teacher_scaling is not None:
            teacher_scaled = teacher_scaled / self.teacher_scaling
        return teacher_scaled

    def fit(self, ins, outs, inspect=False, **kwargs):
        """
        Collect the network's reaction to training data, train readout weights.

        Args:
            ins: np.array of dimensions (N_training_samples x n_inputs)
            outs: np.array of dimension (N_training_samples x n_outputs)
            inspect: show a visualisation of the collected reservoir states
            **kwargs: these arguments are passed to keras' `fit` method

        Returns:
            the network's output on the training data, using the trained weights
        """

        # transform any vectors of shape (x,) into vectors of shape (x,1):
        if ins.ndim < 2:
            ins = np.reshape(ins, (len(ins), -1))
        if outs.ndim < 2:
            outs = np.reshape(outs, (len(outs), -1))
        # transform input and teacher signal:
        teachers_scaled = self._scale_teacher(outs)

        if not self.silent:
            print("harvesting states...")
        # step the reservoir through the given input,output pairs:
        states = np.np.zeros((ins.shape[0], self.n_reservoir))
        for n in range(1, ins.shape[0]):
            states[n, :] = self._update(states[n - 1], ins[n, :],
                                        teachers_scaled[n - 1, :])
        self.states = states

        # learn the weights, i.e. find the linear combination of collected
        # network states that is closest to the target output
        if not self.silent:
            print("fitting...")
        # we'll disregard the first few states:
        transient = min(int(ins.shape[1] / 10), 100)
        # include the raw ins:
        extended_states = np.hstack((states, ins))
        self.extended_states = extended_states

        if self.keras_model is None:
            # Solve for W_out:
            self.W_out = np.dot(np.linalg.pinv(extended_states[transient:, :]),
                                self.inverse_out_activation(teachers_scaled[transient:, :])).T
            # apply learned weights to the collected states:
            pred_train = self._unscale_teacher(self.out_activation(
                np.dot(extended_states, self.W_out.T)))
        else:
            # train the output network on the states
            self.keras_model.fit(extended_states[transient:, :],
                                 teachers_scaled[transient:, :],
                                 **kwargs)
            pred_train = self._unscale_teacher(
                                self.keras_model.predict(extended_states))

        # remember the last state for later:
        self.laststate = states[-1, :]
        self.lastinput = ins[-1, :]
        self.lastoutput = teachers_scaled[-1, :]

        # optionally visualize the collected states
        if inspect:

            plt.figure(
                figsize=(states.shape[0] * 0.0025, states.shape[1] * 0.01))
            plt.imshow(extended_states.T, aspect='auto',
                       interpolation='nearest')
            plt.colorbar()

        if not self.silent:
            print("training error:")
            print(np.sqrt(np.mean((pred_train - outs)**2)))
        return pred_train

    def predict(self, ins, continuation=True):
        """
        Apply the learned weights to the network's reactions to new input.

        Args:
            ins: np.array of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training state

        Returns:
            np.array of output activations
        """

        if ins.ndim < 2:
            ins = np.reshape(ins, (len(ins), -1))
        n_samples = ins.shape[0]

        if continuation:
            laststate = self.laststate
            lastinput = self.lastinput
            lastoutput = self.lastoutput
        else:
            laststate = np.np.zeros(self.n_reservoir)
            lastinput = np.np.zeros(self.n_inputs)
            lastoutput = np.np.zeros(self.n_outputs)

        ins = np.np.vstack([lastinput, ins])
        states = np.np.vstack(
            [laststate, np.np.zeros((n_samples, self.n_reservoir))])
        outs = np.np.vstack(
            [lastoutput, np.np.zeros((n_samples, self.n_outputs))])

        self.extended_states_2 = []
        for n in range(n_samples):
            states[
                n + 1, :] = self._update(states[n, :], ins[n + 1, :], outs[n, :])

            extended_states = np.concatenate([states[n + 1, :],
                                                 ins[n + 1, :]])
            self.extended_states_2.append(extended_states.tolist())
            if self.keras_model is None:
                outs[n + 1, :] = self.out_activation(np.dot(self.W_out,
                                                            extended_states))
            else:
                # keras throws an error if we don't add an empty dimension
                outs[n + 1, :] = self.keras_model.predict(
                    np.expand_dims(extended_states, axis=0))

        self.extended_states_2 = np.mat(self.extended_states_2)

        if self.keras_model is None:
            outs[1:] = self.out_activation(outs[1:])
        return self._unscale_teacher(outs[1:])


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
            print "Running time: Iteratively: {0:.4f}".format(t_f1 - t_i) + \
                  "s \tBlock: {0:.4f}".format(t_f - t_f1) + "s"
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
