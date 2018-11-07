# IMPORTS
import rospy
import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Flatten, Input, LSTM
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import rosbag
from scipy.interpolate import interp1d
import sys
import threading
import time
from tqdm import tqdm

import esn
import processing
import sfa
import utils

# FORCE CPU #
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# MATPLOTLIB STYLE #
plt.style.use('fivethirtyeight')
plt.rc('lines', linewidth=1)
plt.rc('text', usetex=False)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)

nn = [
       [('relu', 40)],
       [('relu', 40)],
       [('relu', 40)]
     ]


class NN(object):

    # ALGORITHM METAPARAMETERS #

    def __init__(self, nn_layers=nn, test_split=0.7, val_split=0.1, stop_delta=0.0001, stop_pat=150,
                 optim='adam', metric='mae', batch_size=2048, max_epochs=10, regularization=0.0,
                 esn_n_res=10, esn_n_read=80, esn_in_mask=None, esn_out_mask=None, esn_real_fb=False,
                 esn_spec_rad=0.3, esn_damping=0.1, esn_sparsity=0.4, esn_noise=0.001,
                 data_file="data/sims/tc.pkl", save_folder="data/nn_learning/", checkpoint=False,
                 no_callbacks=False, verbose=2, random_state=12):

        # ALGORITHM METAPARAMETERS
        self.data_file = data_file
        self.save_folder = save_folder
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.optim = optim
        self.metric = metric
        self.test_split = test_split
        self.val_split = val_split
        self.network_layers = nn_layers
        self.esn_n_read = esn_n_read
        self.esn_real_fb = esn_real_fb
        self.stop_delta = stop_delta
        self.stop_pat = stop_pat
        self.verbose = verbose
        self.save_folder = save_folder
        self.regularization = regularization
        self.checkpoint = checkpoint
        self.no_callbacks = no_callbacks
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # ESN METAPARAMETERS
        self.esn_n_res = esn_n_res
        self.esn_in_mask = esn_in_mask
        self.esn_out_mask = esn_out_mask
        self.esn_spec_rad = esn_spec_rad
        self.esn_damping = esn_damping
        self.esn_sparsity = esn_sparsity
        self.esn_noise = esn_noise

        # ALGO VARIABLES
        self.history = {}
        self.esn = None
        self.readout = None
        self.in_pipe = None
        self.out_pipe = None
        self.predict_in_pipe = None
        self.predict_out_pipe = None
        self.n_in = 0
        self.n_out = 0
        self.t_training_init = None
        self.training_time = 0
        self.epoch = 0
        self.loss = 0
        self.acc = 0
        self.test_loss = 0
        self.test_accuracy = 0
        self.save_index = 0
        self.callbacks = []
        self.graph = None

        # DATA VARIABLES
        self.x_t = None
        self.x2_t = None
        self.y_t = None
        self.x_val = None
        self.x2_val = None
        self.y_val = None
        self.data_n = None
        self.t = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_ft = None
        self.y_ft = None

    # DATA PROCESSING FUNCTIONS #

    def load_data(self, load_all=False, plot_data=False):

        x_new = None
        y_new = None

        if ".bag" in self.data_file:
            # Load the bag file
            x, x2, y = self.load_bag()

            # Format the data
            x_new, y_new = self.format_data(x, x2, y)

            # Save data in numpy array
            with open(self.data_file.replace("bag", "pkl"), "wb") as f:
                pickle.dump([x_new, y_new, self.t], f, protocol=2)

            # Save Interpolation functions in numpy array
            with open(self.data_file.replace(".bag", "_interpol_fct.pkl"), "wb") as f:
                    pickle.dump([self.x_t, self.x2_t, self.y_t, self.x_val, self.x2_val, self.y_val], f, protocol=2)

        if ".pkl" in self.data_file:
            x_new, y_new = self.load_pkl(plot_data)

        # Retrieve all the non-formatted data if load_all flag is enabled
        if load_all:
            names = self.data_file.split(".")
            filename = names[0] + "_interpol_fct.pkl"
            with open(filename, 'rb') as f:
                if sys.version_info[0] >= 3:
                    [self.x_t, self.x2_t, self.y_t,
                     self.x_val, self.x2_val, self.y_val] = pickle.load(f, encoding="latin1")
                else:
                    [self.x_t, self.x2_t, self.y_t,
                     self.x_val, self.x2_val, self.y_val] = pickle.load(f)

        # Split in training and test sets
        self.split_data(x_new, y_new)

    def load_bag(self):

        self.printv("\n\n ===== Collecting Data Bag =====\n")

        x = []
        x2 = []
        y = []

        kadj = "kadj" in self.data_file
        tc = "tc" in self.data_file
        adapt = "adapt" in self.data_file
        prec = "prec" in self.data_file

        # Retrieve the data from the bag file
        bag = rosbag.Bag(self.data_file, 'r')
        for topic, msg, t in tqdm(bag.read_messages()):
            if topic == "/hyq/robot_states":
                r = dict()
                r["t"] = t.to_time()
                r["base"] = []
                r["joint"] = []
                if kadj:
                    r["base"].append(msg.base[0].position)
                    r["base"].append(msg.base[1].position)
                    r["base"].append(msg.base[0].velocity)
                    r["base"].append(msg.base[1].velocity)
                if prec:
                    r["base"].append(msg.base[2].velocity)
                    r["base"].append(msg.base[3].velocity)
                    r["base"].append(msg.base[4].velocity)
                x.append(r)
            if topic == "/hyq/debug":
                r = dict()
                r["t"] = t.to_time()
                stance = []
                vel = []
                if adapt or tc:
                    st = 0
                    for i in range(len(msg.name)):
                        if tc and msg.name[i] in ["forwVel"]:
                            vel = [msg.data[i]]
                        if adapt and st < 4 and msg.name[i] in ["stanceStatusLF", "stanceStatusLH",
                                                                "stanceStatusRF", "stanceStatusRH"]:
                            stance += [msg.data[i]]
                            st += 1

                r["stance"] = stance
                r["vel"] = vel
                x2.append(r)
            if topic == "/hyq/des_joint_states":
                r = dict()
                r["t"] = t.to_time()
                r["pos"] = []
                r["vel"] = []
                if tc:
                    for i in range(len(msg.position)):
                        r["pos"].append(msg.position[i])
                    for i in range(len(msg.velocity)):
                        r["vel"].append(msg.velocity[i])
                y.append(r)

        return x, x2, y

    def load_pkl(self, plot_data=False):

        self.printv("\n\n ===== Collecting Data pkl =====\n")
        with open(self.data_file, "rb") as f:
            if sys.version_info[0] >= 3:
                [x, y, self.t] = pickle.load(f, encoding="latin1")
            else:
                [x, y, self.t] = pickle.load(f)

        if plot_data:
            plt.plot(self.t, x[:, 0:4])
            plt.legend()
            plt.show()
            plt.plot(self.t, y[:, 0:4])
            plt.legend()
            plt.show()

        return x, y

    def get_fct(self, x, y):

        x_new, j = np.unique(x, return_index=True)
        y_new = np.array(y)[j]
        fct = interp1d(x_new, y_new, assume_sorted=True, axis=0)

        return fct

    def interpolate(self, fct, x):

        return fct(x)

    def format_data(self, x, x2, y):

        self.printv("\n\n ===== Formating Data =====")

        # Get values
        self.x_t = np.linspace(0, x[-1]["t"] - x[0]["t"], len(x))      # np.array([r["t"] for r in x]) - x[0]["t"]
        self.x2_t = np.linspace(0, x2[-1]["t"] - x2[0]["t"], len(x2))  # np.array([r["t"] for r in x2]) - x2[0]["t"]
        self.y_t = np.linspace(0, y[-1]["t"] - y[0]["t"], len(y))      # np.array([r["t"] for r in y]) - y[0]["t"]
        self.x_val = np.array([r["joint"] + r["base"] for r in x])
        self.x2_val = np.array([r["stance"] + r["vel"] for r in x2])
        self.y_val = np.array([r["pos"] + r["vel"] for r in y])

        # Get new interpolation range
        self.data_n = max(len(x), len(x2), len(y))

        # Interpolate in new range
        t_min = max(np.min(self.x_t), np.min(self.x2_t), np.min(self.y_t))
        t_max = min(np.max(self.x_t), np.max(self.x2_t), np.max(self.y_t))
        self.t = np.linspace(t_min, t_max, self.data_n)

        x_fct = self.get_fct(self.x_t, self.x_val)
        x2_fct = self.get_fct(self.x2_t, self.x2_val)
        y_fct = self.get_fct(self.y_t, self.y_val)

        x_new = self.interpolate(x_fct, self.t)
        x2_new = self.interpolate(x2_fct, self.t)
        y_new = self.interpolate(y_fct, self.t)

        # Concatenate all input data
        x_new = np.hstack((x_new, x2_new))

        # plt.plot(self.y_t[0:2000], self.y_val[0:2000, 0:3])
        # plt.show()
        # plt.plot(y_new[0:2000, 0:3])
        # plt.show()

        return x_new, y_new

    def split_data(self, x, y):

        self.printv("\n ===== Splitting Data =====\n")

        # Divide into training and testing dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=(1-self.test_split))

        # Print dimensions in a table
        lin = ["NN Controller Inputs", "NN Controller Outputs"]
        col = ["Training", "Validation", "Testing"]
        dat = [[(int(x_train.shape[0]*(1-self.val_split)), x_train.shape[1]),
                (int(y_train.shape[0]*(1-self.val_split)), y_train.shape[1])],
               [(int(x_train.shape[0]*self.val_split), x_train.shape[1]),
                (int(y_train.shape[0]*self.val_split), y_train.shape[1])],
               [x_test.shape, y_test.shape]]
        row_format = "{:>25}" * (len(lin) + 1)
        # self.printv(row_format.format("", *lin))
        # for c, r in zip(col, dat):
        #     self.printv(row_format.format(c, *r))

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_data_to_save(self):

        utils.make_keras_picklable()

        f_in_folder = [f for f in os.listdir(self.save_folder)
                       if os.path.isfile(os.path.join(self.save_folder, f))]

        if not hasattr(self, 'save_index'):
            if len([s for s in f_in_folder if "network" in s]) == 0:
                self.save_index = 0
            else:
                self.save_index = max([int(utils.split(f, "_.")[-2])
                                       for f in [s for s in f_in_folder if "network" in s]]) + 1
        else:
            self.save_index = 0

        # Save training data
        to_save = copy.copy(self.__dict__)
        if 'esn' in to_save.keys():
            del to_save["esn"]
        if 'readout' in to_save.keys():
            del to_save["readout"]
        if 'callbacks' in to_save.keys():
            for c in to_save["callbacks"]:
                if hasattr(c, 'model'):
                    del c.model
        if 'x_train' in to_save.keys():
            del to_save["x_train"], to_save["y_train"]
            del to_save["x_test"], to_save["y_test"]

        return [self.save_folder, self.save_index, self.__dict__]

    # ELEMENTARY PIPE FUNCTIONS #

    def get_in_dim(self):

        # Create dummy data to get pipe length
        dummy = np.zeros(self.x_train.shape)
        y = self.in_pipe.transform(dummy)

        return y.shape

    def create_x_scaler(self):

        return processing.HyQStateScaler()
        # return MinMaxScaler((-1, 1))

    def create_y_scaler(self):

        return processing.HyQJointScaler()
        # return MinMaxScaler((-1, 1))

    def create_callbacks(self):

        if not self.no_callbacks:
            # Wrap in a regressor
            if not hasattr(self, 'callbacks'):
                self.callbacks = []

            cb_names = [c.__class__.__name__ for c in self.callbacks]
            if "NetworkSave" not in cb_names and self.checkpoint:
                self.callbacks += [utils.NetworkSave(self.get_data_to_save(),
                                                     monitor='val_loss', verbose=self.verbose,
                                                     save_best_only=True, mode='min',
                                                     max_epochs=self.max_epochs)]

            if "CustomHistory" not in cb_names:
                self.callbacks += [utils.CustomHistory()]

            if "TensorBoard" not in cb_names:
                self.callbacks += [TensorBoard()]

            if "EarlyStopping" not in cb_names:
                if self.stop_delta is not None:
                    self.callbacks += [EarlyStopping(monitor='val_loss',
                                                     mode="min",
                                                     verbose=self.verbose,
                                                     patience=self.stop_pat,
                                                     min_delta=self.stop_delta)]

            if "PlotJupyter" not in cb_names:
                if self.verbose == -1:
                    plt.rc('figure', autolayout=False)
                    utils.make_keras_picklable()
                    self.callbacks += [utils.PlotJupyter(self.x_test[:400, :], self.y_test[:400, :], network=self,
                                                         plot=False, max_epochs=self.max_epochs, verbose=self.verbose)]
                if self.verbose == -2 or self.verbose == -3:
                    plt.rc('figure', autolayout=False)
                    utils.make_keras_picklable()
                    self.callbacks += [utils.PlotJupyter(self.x_ft[:400, :], self.y_ft[:400, :], network=self,
                                                         plot=True, max_epochs=self.max_epochs, verbose=self.verbose)]
        verbose = self.verbose
        if verbose > 1:
            verbose = 1

        return verbose

    def create_readout_fn(self):

        # Input Layer
        n_in = self.esn_n_read + sum([not i for i in self.esn_in_mask])
        state_input = Input(shape=(n_in, 1, ),
                            name='robot_state')
        x = Flatten()(state_input)

        # Network layers
        for l in self.network_layers:
            if l[0][0] in ['relu', 'tanh', 'linear']:
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
        readout_fn = Model(inputs=[state_input], outputs=x)
        keras.losses.custom_loss = self.metric
        readout_fn.compile(loss=keras.losses.custom_loss,
                           optimizer=self.optim,
                           metrics=['accuracy', 'mse', 'mae', 'mape', 'cosine'])
        if self.verbose > 1:  # or self.verbose == -1:
            readout_fn.summary()

        # Make the model graph thread-safe
        self.graph = tf.get_default_graph()

        return readout_fn

    def create_nn(self):

        self.printv("\n\n ===== Creating Network =====")

        self.create_callbacks()

        # Create the Keras model
        if self.esn_real_fb:
            # We are not using the fit function, only train_on_batch
            self.readout = self.create_readout_fn()
        else:
            if self.max_epochs != 0:
                self.readout = self.create_readout_fn()

            else:
                self.readout = None

        # Create a random reservoir with bias
        self.esn = esn.FeedbackESN(n_inputs=self.n_in, n_outputs=self.n_out, n_read=self.esn_n_read,
                                   in_esn_mask=self.esn_in_mask, out_esn_mask=self.esn_out_mask,
                                   n_reservoir=self.esn_n_res, spectral_radius=self.esn_spec_rad,
                                   damping=self.esn_damping, sparsity=self.esn_sparsity, noise=self.esn_noise,
                                   fb_scaling=1, fb_shift=0, keras_model=self.readout, real_fb=self.esn_real_fb,
                                   random_state=self.random_state, verbose=self.verbose)

        return self.esn

    def create_pp(self):

        pp_pipe = []
        i = 0

        for l in self.network_layers:
            if len(l) > 1:
                tl = []
                tw = {}
                for k in l:
                    step_name = None
                    e = None
                    if k[0] == 'esnsfa':
                        e = sfa.ESNSFA(n_readout=self.esn_n_read, n_components=k[1], weight_scaling=k[2],
                                       n_res=k[3], l_min=k[4], l_max=k[5])
                        step_name = 'esnsfa' + str(i)

                    if k[0] == 'td':
                        e = processing.TimeDelay(k[1], k[2])
                        step_name = 'td' + str(i)

                    if k[0] == 'osc':
                        e = processing.Oscillator(t=self.t, r=self.test_split)
                        step_name = 'osc' + str(i)

                    if k[0] == 'fft':
                        e = processing.FFT(self.t[1] - self.t[0], k[1])
                        step_name = 'fft' + str(i)

                    if k[0] == 'sdec':
                        e = processing.SeasonalDecomposition(self.t[1] - self.t[0])
                        step_name = 'sdec' + str(i)

                    if k[0] == 'noise':
                        e = processing.GaussianNoise(stdev=k[1])
                        step_name = 'noise' + str(i)

                    if step_name is not None and e is not None:
                        tl += [(step_name, e)]
                        tw[step_name] = 1
                pp_pipe += [('fu', FeatureUnion(transformer_list=tl,
                                                transformer_weights=tw))]
            else:
                if l[0][0] == 'esnsfa':
                    e = sfa.ESNSFA(n_readout=self.esn_n_read,  n_components=l[0][1], weight_scaling=l[0][2],
                                   n_res=l[0][3], l_min=l[0][4], l_max=l[0][5])
                    pp_pipe += [('esnsfa' + str(i), e)]

                if l[0][0] == 'fft':
                    e = processing.FFT(self.t[1] - self.t[0], l[0][1])
                    pp_pipe += [('fft' + str(i), e)]

                if l[0][0] == 'sdec':
                    e = processing.SeasonalDecomposition(self.t[1] - self.t[0])
                    pp_pipe += [('sdec' + str(i), e)]

                if l[0][0] == 'td':
                    e = processing.TimeDelay(l[0][1], l[0][2])
                    pp_pipe += [('td' + str(i), e)]

                if l[0][0] == 'osc':
                    e = processing.Oscillator(t=self.t, r=self.test_split)
                    pp_pipe += [('osc' + str(i), e)]

                if l[0][0] == 'noise':
                    e = processing.GaussianNoise(stdev=l[0][1])
                    pp_pipe += [('noise' + str(i), e)]

            i += 1

        return pp_pipe

    # PLOT AND EVALUATION FUNCTIONS #

    def save(self):

        self.printv("\n\n ===== Saving =====\n")

        # Get the index of the saved file
        [foldername, index, to_save] = self.get_data_to_save()

        # Save ESN + Readout
        self.esn.save(foldername, index)

        # Remove all unpickable data and save
        to_save = copy.copy(to_save)
        if 'esn' in to_save.keys():
            del to_save["esn"]
        if 'readout' in to_save.keys():
            del to_save["readout"]
        if 'callbacks' in to_save.keys():
            del to_save["callbacks"]
        pickle.dump(to_save, open(foldername + "/nn_" + str(index) + ".pkl", "wb"), protocol=2)
        del to_save

        self.printv("Files saved: " + self.save_folder + "/readout_" + str(index) + ".h5, nn_" + str(index) +
                    ".pkl, nn_" + str(index) + ".pkl")

    def load(self, foldername, num=0, load_all=False):

        # Load this class
        verbose_rem = self.verbose
        with open(foldername + "/nn_" + str(num) + ".pkl", 'rb') as f:
            if sys.version_info[0] >= 3:
                self.__dict__ = pickle.load(f, encoding='latin1')
            else:
                self.__dict__ = pickle.load(f)

            self.verbose = verbose_rem

        # Re-create a history
        utils.history = self.history

        # Load the esn and the readout
        self.esn = esn.FeedbackESN(1, 1, [False], [True])
        self.esn.load(foldername, num)
        self.readout = self.esn.keras_model

        # Load the callbacks
        if 'callbacks' in self.__dict__.keys():
            for c in self.callbacks:
                c.model = self.esn.keras_model

        # Load dataset
        self.load_data(load_all)

    def printv(self, txt):

        if self.verbose >= 1:
            print(txt)

    def plot_histogram(self, x):

        plt.hist(x.flatten(), 50, facecolor='g', alpha=0.75)

        plt.figure()
        plt.xlabel('Input values')
        plt.ylabel('Distribution')
        plt.grid(True)
        plt.show()

    def plot_fft(self):

        self.printv("\n\n ===== Computing FFT =====\n")

        # Process features
        x_ft = self.in_pipe.transform(self.x_test)
        self.out_pipe.transform(self.y_test)

        # Process NN
        y_pred = self.out_pipe.inverse_transform(self.esn.predict(x_ft, batch_size=self.batch_size,
                                                                  verbose=self.verbose))

        y_truth = self.y_test

        ts = self.t[1] - self.t[0]
        fs = 1.0/ts

        n = y_truth.shape[0]
        k = np.arange(n)
        period = n / fs
        frq = k / period
        frq = frq[range(n/2)]

        y_pred_fft = np.fft.fft(y_pred[:, 0]) / n
        y_pred_fft = y_pred_fft[range(n/2)]
        y_truth_fft = np.fft.fft(y_truth[:, 0]) / n
        y_truth_fft = y_truth_fft[range(n/2)]

        plt.plot(frq, abs(y_pred_fft), label="Predicted joints FFT")
        plt.plot(frq, abs(y_truth_fft), label="Test joints FFT")
        plt.xlabel('Freq (Hz)')
        plt.ylabel('|Y(freq)|')
        plt.xlim([0.05, max(frq)])
        plt.ylim([0, max(y_pred_fft[2:] + 0.002)])
        plt.legend()
        plt.show()

    def plot_hist(self):

        h = self.history
        if "loss" in h and "val_loss" in h:
            plt.figure()
            plt.plot(h['loss'][:self.max_epochs])
            plt.plot(h['val_loss'][:self.max_epochs])
            plt.title('Model Loss')
            plt.ylabel('Loss [MAE]')
            plt.xlabel('Epoch [#]')
            plt.legend(['Training', 'Validation'], loc='upper left')
            plt.show()

    def plot_pred(self, t, y_pred, y_truth , plot_relation=False):

        if plot_relation:
            plt.figure()
            plt.plot(y_truth, y_pred, marker='o', linestyle='None')
            plt.xlabel("Actual Position or Speed [rad - rad/s]")
            plt.ylabel("Predicted Position or Speed [rad - rad/s]")
            plt.show()

        plt.figure()
        plt.plot(t, y_truth, label="Target")
        plt.plot(t, y_pred, label="Prediction")
        # plt.plot(t, np.abs(y_truth - y_pred), label="MAE error")
        plt.xlabel("Time [s]")
        plt.ylabel("Position or Speed [rad - rad/s]")
        plt.legend()
        plt.show()

    def evaluate(self, plot_states=False, plot_test=False, plot_hist=False, plot_histogram=False,
                 plot_fft=False, plot_relation=False, win=5000):

        self.printv("\n\n ===== Evaluating Test Dataset =====\n")

        # Process features
        y_truth = self.y_test
        x_ft = self.transform_x_ft(self.x_test)
        y_ft = self.transform_y_ft(y_truth)

        # Process NN
        y_pred_ft = self.esn.predict(x_ft, inspect=plot_states, win=win, batch_size=self.batch_size, verbose=0)
        y_pred = self.out_pipe.inverse_transform(y_pred_ft)

        # Display score
        score = np.sqrt(np.mean((y_pred_ft - y_ft)**2))
        self.printv("Test loss: " + str(score))
        self.test_loss = score

        # Plot
        if plot_histogram:
            self.plot_histogram(x_ft[:, :, 0])
        if plot_fft:
            self.plot_fft()
        if plot_hist:
            self.plot_hist()
        if plot_test:
            t = self.t[0:win]
            self.plot_pred(t, y_pred[:win, 0:2], y_truth[:win, 0:2], plot_relation=plot_relation)

        return y_truth, y_pred, score

    # MAIN FUNCTIONS #

    def fit_transform_ft(self, x, y):

        # Create IN pipe and transform
        self.printv("\n\n ===== Transform In Features =====")

        self.in_pipe = Pipeline([('xsc0', self.create_x_scaler())] + self.create_pp())
        self.x_ft = self.in_pipe.fit_transform(x)

        # Create OUT pipe and transform
        self.printv("\n\n ===== Transform Out Features =====")
        self.out_pipe = Pipeline([('ysc', self.create_y_scaler())])
        self.y_ft = self.out_pipe.fit_transform(y)

        # Count and print the number of inputs/outputs
        self.n_in = self.x_ft.shape[1]
        self.n_out = y.shape[1]
        self.printv("\nTotal input features size: " + str(self.x_ft.shape))
        self.printv("Total output features size: " + str(y.shape))

        # Create ESN masks
        if self.esn_in_mask is None:
            self.esn_in_mask = [False for _ in range(self.n_in)]
        if self.esn_out_mask is None:
            self.esn_out_mask = [False for _ in range(self.n_out)]

        return self.x_ft, self.y_ft

    def transform_x_ft(self, x):

        # We create a second pipe for prediction to keep it thread-safe when we also train
        if self.predict_in_pipe is None:
            self.predict_in_pipe = Pipeline([('xsc0', self.create_x_scaler())] + self.create_pp())

        x_ft = self.predict_in_pipe.transform(x)
        return x_ft

    def transform_y_ft(self, y):

        # We create a second pipe for prediction to keep it thread-safe when we also train
        if self.predict_out_pipe is None:
            self.predict_out_pipe = Pipeline([('ysc0', self.create_y_scaler())])

        self.y_ft = self.predict_out_pipe.transform(y)
        return self.y_ft

    def inverse_transform_y_ft(self, y_ft):

        # We create a second pipe for prediction to keep it thread-safe when we also train
        if self.predict_out_pipe is None:
            self.predict_out_pipe = Pipeline([('ysc', self.create_y_scaler())])

        return self.predict_out_pipe.inverse_transform(y_ft)

    def predict(self, x):

        x_ft = self.transform_x_ft(x)
        with self.graph.as_default():
            y_ft = self.esn.predict(x_ft, batch_size=self.batch_size, verbose=0)
        y = self.inverse_transform_y_ft(y_ft)

        return y

    def train(self, x=None, y=None, n_epochs=None, evaluate=True, plot_test_states=False, plot_train_states=False,
              plot_data=False, plot_fft=False, plot_hist=False, plot_train=False, plot_test=False,
              plot_relation=False, win=5000, save=True):

        # Determine the number of training epochs
        if n_epochs is None:
            tot_epochs = self.max_epochs
        else:
            tot_epochs = self.epoch + n_epochs
            if tot_epochs > self.max_epochs:
                tot_epochs = self.max_epochs

        # For the first iteration
        if self.epoch == 0:
            # If no x and y are specfied, use the loaded training set
            if x is None:
                self.load_data(plot_data=plot_data)
                x = self.x_train
            if y is None:
                y = self.y_train

            self.t_training_init = time.time()

            # Get features and fit the estimators
            x_ft, y_ft = self.fit_transform_ft(x, y)

            # Create the network
            self.esn = self.create_nn()
        else:
            # If no x and y are specfied, use the loaded training set
            if x is None:
                x = self.x_train
            if y is None:
                y = self.y_train

            # Get features and fit the network
            x_ft, y_ft = self.fit_transform_ft(x, y)

        # Fit the network
        self.printv("\n\n ===== Training Network =====\n")
        if self.epoch == 0:
            verbose = self.create_callbacks()
            with self.graph.as_default():
                y_train_pred = self.esn.fit(x_ft, y_ft, inspect=plot_train_states, epochs=tot_epochs,
                                            validation_split=self.val_split, batch_size=self.batch_size,
                                            callbacks=self.callbacks, verbose=verbose)

        else:
            if n_epochs != 1:
                verbose = self.create_callbacks()
                with self.graph.as_default():
                    y_train_pred = self.esn.fit(x_ft, y_ft, inspect=plot_train_states, state_update=True,
                                                epochs=tot_epochs, validation_split=self.val_split,
                                                callbacks=self.callbacks, initial_epoch=self.epoch,
                                                batch_size=self.batch_size, verbose=verbose)
            else:
                self.create_callbacks()
                with self.graph.as_default():
                    y_train_pred = self.esn.fit(x_ft, y_ft, inspect=plot_train_states, callbacks=self.callbacks,
                                                epochs=tot_epochs, validation_split=self.val_split,
                                                batch_size=self.batch_size, verbose=verbose)

        self.epoch = tot_epochs
        self.training_time = time.time() - self.t_training_init
        self.history = utils.history

        # Save all
        if save:
            self.save()

        # Show plots
        if plot_train:
            t_max = self.y_ft.shape[0]
            self.plot_pred(self.t[t_max-win:t_max], y_train_pred[-win:, 1],
                           self.y_ft[-win:, 1], plot_relation=plot_relation)
        if plot_hist:
            self.plot_hist()

        # Show Evaluation
        if evaluate:
            self.evaluate(plot_states=plot_test_states, plot_hist=False, plot_test=plot_test, plot_fft=plot_fft,
                          plot_relation=plot_relation, plot_histogram=False, win=win)
            return self.test_loss, self.test_accuracy
        else:
            if {"val_loss", "val_acc"} in set(self.history):
                return self.history['val_loss'][-1], \
                       self.history['val_acc'][-1]
            else:
                return 1, 0


if __name__ == '__main__':

    if len(sys.argv) > 1:

        if sys.argv[1] == "process":
            nn = NN(data_file=sys.argv[2])
            nn.load_data()

        if sys.argv[1] == "test":
            nn = NN()
            nn.load(sys.argv[2])
            nn.evaluate(plot_states=True, plot_test=True, win=10000)

        if sys.argv[1] == "continue":
            nn = NN()
            nn.load(sys.argv[2])
            nn.train()

        if sys.argv[1] == "train":

            folder = "data/nn_learning/" + utils.timestamp()
            utils.mkdir(folder)
            nn = NN(data_file=sys.argv[2], save_folder=folder)
            nn.train(plot_train=True, plot_train_states=True, plot_test=True, plot_test_states=True)
