# IMPORTS

import keras
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Flatten, Input, LSTM
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

import copy
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
       # [('noise', 0.1)],
       # [('sdec',)],
       # [('esn', 100, 0.9)],
       # [('sdec',), ('td', 10, 2), ('esnsfa', 50, 0.9, 14, -3, 0)],
       # [('td', 6, 2)],#,
       # [('esnsfa', 50, 0.9, 14, -3, 0)],
       # [('fft', 12)],
       # [('esnsfa', 80, 0.9, 15, -2, 0), ('td', 10, 2)], #
       # [('lstm', 30)],
       # [('noise', 200, 0.01)],
       # [('relu', 40)],
       # [('relu', 40)],
       [('osc',)],
       [('relu', 40)]
       # [('relu', 20)],
     ]


class NN(object):

    # ALGORITHM METAPARAMETERS #

    def __init__(self, batch_size=2048, max_epochs=1000, test_split=0.7,
                 val_split=0.1, verbose=1, stop_pat=150, checkpoint=True,
                 nn_layers=nn, stop_delta=0.0001, esn_n_read=24, optim='adam',
                 metric='mae', data_file="data/sims/simple_walk.bag",
                 regularization=0.001, save_folder="data/nn_learning/"):

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
        self.stop_delta = stop_delta
        self.stop_pat = stop_pat
        self.verbose = verbose
        self.save_folder = save_folder
        self.regularization = regularization
        self.checkpoint = checkpoint

        self.history = {}
        self.nn = None
        self.in_pipe = None
        self.out_pipe = None
        self.predict_in_pipe = None
        self.predict_out_pipe = None
        self.training_time = 0
        self.epoch = 0
        self.loss = 0
        self.acc = 0
        self.test_loss = 0
        self.test_accuracy = 0
        self.save_index = 0
        self.callbacks = []

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
        self.n_in = 0
        self.n_out = 0
        self.x_ft = None
        self.t_training_init = None

    # DATA PROCESSING FUNCTIONS #

    def load_data(self, load_all=False):

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
            x_new, y_new = self.load_pkl()

        # Retrieve all the non-formatted data if load_all flag is enabled
        if load_all:
            names = self.data_file.split(".")
            filename = names[0] + "_interpol_fct.pkl"
            with open(filename, 'rb') as f:
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

    def load_pkl(self):

        self.printv("\n\n ===== Collecting Data pkl =====\n")

        with open(self.data_file, "rb") as f:
            [x, y, self.t] = pickle.load(f)

        plt.plot(x)
        plt.legend()
        plt.show()

        return x, y

    def get_fct(self, x, y):

        x_new, j = np.unique(x, return_index=True)
        y_new = np.array(y)[j]
        fct = interp1d(x_new, y_new, assume_sorted=False, axis=0)

        return fct

    def interpolate(self, fct, x):

        return fct(x)

    def format_data(self, x, x2, y):

        self.printv("\n\n ===== Formating Data =====")

        # Get values
        self.x_t = np.array([r["t"] for r in x]) - x[0]["t"]
        self.x2_t = np.array([r["t"] for r in x2]) - x2[0]["t"]
        self.y_t = np.array([r["t"] for r in y]) - y[0]["t"]
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
        self.printv(row_format.format("", *lin))
        for c, r in zip(col, dat):
            self.printv(row_format.format(c, *r))

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
        if 'nn' in to_save.keys():
            del to_save["nn"]
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

        return processing.HyQStateScaler(n_in=self.x_train.shape[1])
        # return MinMaxScaler((-1, 1))

    def create_y_scaler(self):

        return processing.HyQJointScaler()
        # return MinMaxScaler((-1, 1))

    def create_callbacks(self):

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

        if "PlotJupyter" not in cb_names and self.verbose == -1:
            self.callbacks += [utils.PlotJupyter(self.x_test[:400, :], self.y_test[:400, :], self)]

        verbose = self.verbose
        if verbose > 1:
            verbose = 1

        return verbose

    def create_nn_fn(self):

        # Input Layer
        n_in = self.n_in
        state_input = Input(shape=(n_in, 1, ),
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
        # x = Activation('tanh')(x)

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

        self.printv("\n\n ===== Creating Network =====")

        verbose = self.create_callbacks()
        return KerasRegressor(build_fn=self.create_nn_fn,
                              validation_split=self.val_split,
                              batch_size=self.batch_size,
                              epochs=self.max_epochs,
                              callbacks=self.callbacks,
                              verbose=max(verbose, 0))

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
                    if k[0] == 'esn':
                        e = esn.ForwardESN(n_readout=self.esn_n_read, n_components=k[1], weight_scaling=k[2])
                        step_name = 'esn' + str(i)

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
                if l[0][0] == 'esn':
                    e = esn.ForwardESN(n_readout=self.esn_n_read, n_components=l[0][1], weight_scaling=l[0][2])
                    pp_pipe += [('esn' + str(i), e)]

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

        utils.make_keras_picklable()

        self.printv("\n\n ===== Saving =====\n")

        f_in_folder = [f for f in os.listdir(self.save_folder)
                       if os.path.isfile(os.path.join(self.save_folder, f))]

        if len([s for s in f_in_folder if "network" in s]) == 0:
            index = 0
        else:
            index = max([int(utils.split(f, "_.")[-2]) for f in [s for s in f_in_folder if "network" in s]]) + 1

        # Save training data
        to_save = copy.copy(self.__dict__)
        del to_save["nn"], to_save["callbacks"][1].model,
        del to_save["x_train"], to_save["y_train"]
        del to_save["x_test"], to_save["y_test"]
        pickle.dump(to_save, open(self.save_folder + "/network_" + str(index) + ".pkl", "wb"), protocol=2)
        del to_save

        # Save nn model
        if self.nn is not None:
            self.nn.save(self.save_folder + "/model_" + str(index) + ".h5")

        self.printv("Files saved: " + self.save_folder + "/model_" + str(index) + ".h5, pipe_" + str(index) +
                    ".pkl, network_" + str(index) + ".pkl")

    def load(self, foldername, num=0, load_all=False):

        # Load this class
        verbose_rem = self.verbose
        with open(foldername + "/network_" + str(num) + ".pkl", 'rb') as f:
            self.__dict__ = pickle.load(f)
            self.verbose = verbose_rem

        utils.history = self.history

        # Load network
        self.nn = load_model(foldername + "/model_" + str(num) + ".h5")

        if 'callbacks' in self.__dict__.keys():
            for c in self.callbacks:
                c.model = load_model(foldername + "/model_" + str(num) + ".h5")

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
        x_ft = np.expand_dims(x_ft, axis=2)
        self.out_pipe.transform(self.y_test)

        # Process NN
        y_pred = self.out_pipe.inverse_transform(self.nn.predict(x_ft, batch_size=self.batch_size,
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

    def evaluate(self, show=True):

        self.printv("\n\n ===== Evaluating Test Dataset =====\n")

        # Process features

        x_ft = self.in_pipe.transform(self.x_test)
        x_ft = np.expand_dims(x_ft, axis=2)
        y_ft = self.out_pipe.transform(self.y_test)

        # self.plot_histogram(x_ft[:, :, 0])

        # Process NN
        score = self.nn.evaluate(x_ft, y_ft, verbose=2)
        y_pred_ft = self.nn.predict(x_ft, batch_size=self.batch_size, verbose=0)
        y_pred = self.out_pipe.inverse_transform(y_pred_ft)

        y_truth = self.y_test
        self.printv("Test loss: " + str(score[0]))
        self.printv("Test accuracy: " + str(score[1]))
        self.test_loss = score[0]
        self.test_accuracy = score[1]

        if show:
            # Summarize history for loss
            h = self.history
            plt.figure()
            plt.plot(h['loss'][:self.max_epochs])
            plt.plot(h['val_loss'][:self.max_epochs])
            plt.title('Model Loss')
            plt.ylabel('Loss [MAE]')
            plt.xlabel('Epoch [#]')
            plt.legend(['Training', 'Validation'], loc='upper left')
            plt.show()

            # Plot test and predicted values
            plt.figure()
            plt.plot(y_truth[:, 0], y_pred[:, 0],
                     marker='o', linestyle='None')
            plt.xlabel("FL HAA Actual Position [rad]")
            plt.ylabel("FL HAA Predicted Position [rad]")
            plt.show()
            plt.figure()
            t = self.t[0:y_truth.shape[0]]
            plt.plot(t, y_truth[:, 0], label="Real")
            plt.plot(t, y_pred[:, 0], label="Predicted")
            plt.plot(t, np.abs(y_truth - y_pred)[:, 0],
                     label="MAE error")
            plt.xlabel("Time [s]")
            plt.ylabel("FL HAA Position [rad]")
            plt.legend()
            plt.show()

            # ts = [0]
            # for i in range(len(self.t) - 1):
            #     ts.append(self.t[i+1] - self.t[i])
            # plt.plot(self.t, ts)
            # plt.show()

        return y_truth, y_pred, score

    # MAIN FUNCTIONS #

    def fit_transform_ft(self, x, y, show=True):

        # Create IN pipe and transform
        self.printv("\n\n ===== Transform In Features =====")

        pp_pipe = self.create_pp()
        if len(pp_pipe) > 0:
            if "esn" in pp_pipe[0][0]:
                print("WARNING: when using a ESN, the input is the prediction output and not the observed state!" +
                      " Training can only be done step by step!!")
                self.in_pipe = Pipeline([('xsc0', self.create_y_scaler())] + pp_pipe)
            else:
                self.in_pipe = Pipeline([('xsc0', self.create_x_scaler())] + pp_pipe)
        else:
            self.in_pipe = Pipeline([('xsc0', self.create_x_scaler())])
        x_ft = self.in_pipe.fit_transform(x)
        self.x_ft = x_ft
        x_ft = np.expand_dims(x_ft, axis=2)

        self.n_in = x_ft.shape[1]
        self.n_out = y.shape[1]
        self.printv("\nTotal input features size: " + str(x_ft.shape))
        self.printv("Total output features size: " + str(y.shape))

        flatten_nn_struct = utils.flatten(self.in_pipe.steps)
        if show:
            for e in flatten_nn_struct:
                if isinstance(e, tuple):
                    if 'esnsfa' in e[0]:
                        ts = self.t[1] - self.t[0]
                        t_max = e[1].y.shape[0] * ts
                        print(ts, t_max)
                        t = np.arange(0, t_max, ts)
                        e[1].plot_sfa(t)

        # Create OUT pipe and transform
        self.printv("\n\n ===== Transform Out Features =====")
        raw_pipe = [('ysc', self.create_y_scaler())]
        self.out_pipe = Pipeline(raw_pipe)
        y_ft = self.out_pipe.fit_transform(y)

        return x_ft, y_ft

    def predict(self, state):

        # We create a second pipe for prediction to keep it thread-safe
        # when we also train
        if self.predict_in_pipe is None:
            self.predict_in_pipe = \
                    Pipeline([('xsc0', self.create_x_scaler())] + self.create_pp())
            self.predict_out_pipe = Pipeline([('ysc', self.create_y_scaler())])

        if len(self.predict_in_pipe.steps) > 1:
            if "esn" in self.predict_in_pipe.steps[1][0]:
                x_ft = state
                x_ft = np.expand_dims(x_ft, axis=2)
                y_ft = self.nn.predict_on_batch(x_ft)
                y = y_ft
                return y

        x_ft = self.predict_in_pipe.transform(state)
        self.x_ft = x_ft
        x_ft = np.expand_dims(x_ft, axis=2)
        y_ft = self.nn.predict_on_batch(x_ft)
        y = self.predict_out_pipe.inverse_transform(y_ft)
        # return y[0,:].flatten()
        return y

    def train(self, x=None, y=None, show=True, n_epochs=None, evaluate=True):

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
                self.load_data()
                x = self.x_train  # [self.epoch:tot_epochs]
            if y is None:
                y = self.y_train  # [self.epoch:tot_epochs]

            # Create the network
            self.nn = self.create_nn()
            self.t_training_init = time.time()

            # Get features and fit the estimators
            x_ft, y_ft = self.fit_transform_ft(x, y, show)
        else:
            # If no x and y are specfied, use the loaded training set
            if x is None:
                x = self.x_train  # [self.epoch:tot_epochs]
            if y is None:
                y = self.y_train  # [self.epoch:tot_epochs]

            # Get features and fit the network
            with threading.Lock():
                x_ft = self.in_pipe.transform(x)
                x_ft = np.expand_dims(x_ft, axis=2)
                y_ft = self.out_pipe.transform(y)

        # Fit the network
        self.printv("\n\n ===== Training Network =====\n")
        if self.epoch == 0:
            self.nn.fit(x_ft, y_ft, epochs=tot_epochs)

        else:
            if n_epochs != 1:
                verbose = self.create_callbacks()
                self.nn.fit(x_ft, y_ft, validation_split=self.val_split, epochs=tot_epochs, initial_epoch=self.epoch,
                            callbacks=self.callbacks, batch_size=self.batch_size, verbose=verbose)
            else:
                self.create_callbacks()
                self.nn.train_on_batch(x_ft, y_ft)

        self.epoch = tot_epochs
        self.training_time = time.time() - self.t_training_init
        self.history = utils.history

        # Show Evaluation
        if evaluate:
            self.evaluate(show)
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
            nn.evaluate(True)

        if sys.argv[1] == "continue":
            nn = NN()
            nn.load(sys.argv[2])
            nn.train(True)

        if sys.argv[1] == "fft":
            nn = NN()
            nn.load(sys.argv[2])
            nn.plot_fft()

        if sys.argv[1] == "train":

            folder = "data/nn_learning/" + utils.timestamp()
            utils.mkdir(folder)
            nn = NN(data_file=sys.argv[2], save_folder=folder)
            nn.train(show=False)
