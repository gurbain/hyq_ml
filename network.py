## IMPORTS

from keras.callbacks import EarlyStopping, ModelCheckpoint, History, TensorBoard
from keras.losses import categorical_crossentropy, mean_absolute_error, mean_squared_error
from keras.optimizers import Adadelta, Adam
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, GaussianNoise
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re
import rospy as ros
import rosbag
from scipy.interpolate import interp1d
import sys
import threading
import time
from tqdm import tqdm

import esn
import physics
import utils

## FORCE CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

## MATPLOTLIB STYLE
plt.style.use('fivethirtyeight')
plt.rc('lines', linewidth= 1)
plt.rc('text', usetex=False)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)

nn = [
       #[('noise', 0.1)],
       [('td', 6, 1)],#,
       #[('esnsfa', 50, 0.9, 8, -3, 0)], #
       #[('noise', 200, 0.01)],
       [('relu', 4096)],
       [('relu', 2048)],
       [('relu', 512)],
     ]


class FeedForwardNN():

    ## ALGORITHM METAPARAMETERS

    def __init__(self, batch_size=2048, max_epochs=100, test_split=0.7,
                 val_split=0.1, verbose=2, stop_pat=40,
                 nn_layers=nn, stop_delta=0.0001, esn_n_read=10,
                 data_file="data/sims/simple_walk.bag",
                 regularization=0.001, save_folder="data/nn_learning/"):

        ## ALGORITHM METAPARAMETERS
        self.data_file = data_file
        self.save_folder = save_folder
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.test_split = test_split
        self.val_split = val_split
        self.network_layers = nn_layers
        self.esn_n_read = esn_n_read
        self.stop_delta = stop_delta
        self.stop_pat = stop_pat
        self.verbose = verbose
        self.save_folder = save_folder
        self.regularization = regularization

        self.history = None
        self.nn = None
        self.nn_pipe = None
        self.training_time = 0
        self.train_it = 0
        self.loss = 0
        self.acc = 0

    ## DATA PROCESSING FUNCTIONS

    def load_data(self, load_all=False):

        if ".bag" in self.data_file:
            # Load the bag file
            x, x2, y = self.load_bag()

            # Format the data
            x_new, y_new = self.format_data(x, x2, y)

            # Save data in numpy array
            with open(self.data_file.replace("bag", "pkl") , "wb") as f:
                pickle.dump([x_new, y_new, self.t], f, protocol=2)

            # Save Interpolation functins in numpy array
            with open(self.data_file.replace(".bag",
                "_interpol_fct.pkl") , "wb") as f:
                    pickle.dump([self.x_t, self.x2_t, self.y_t,
                                 self.x_val, self.x2_val, self.y_val],
                                 f, protocol=2)


        if ".pkl" in self.data_file:
            x_new, y_new = self.load_pkl()

        # Retrieve all the non-formatted data if load_all flag is enabled
        if load_all:
            names = self.data_file.split(".")
            file = names[0] + "_interpol_fct.pkl"
            with open(file,'rb') as f:
                [self.x_t, self.x2_t, self.y_t,
                 self.x_val, self.x2_val, self.y_val] = pickle.load(f)

        # Split in training and test sets
        self.split_data(x_new, y_new)

    def load_bag(self):

        self.printv("\n\n ===== Collecting Data Bag =====\n")

        x = []
        x2 = []
        y = []
        fd = []

        # Retrieve the data from the bag file
        bag = rosbag.Bag(self.data_file, 'r')
        t_init = bag.get_start_time()
        for topic, msg, t in tqdm(bag.read_messages()):
            if topic == "/hyq/robot_states":
                r = dict()
                r["t"] = t.to_time()
                r["base"] = []
                for i in range(len(msg.base)):
                    r["base"].append(msg.base[i].velocity)
                r["joint"] = []
                for i in range(len(msg.joints)):
                    r["base"].append(msg.joints[i].position)
                for i in range(len(msg.joints)):
                    r["base"].append(msg.joints[i].velocity)
                for i in range(len(msg.joints)):
                    r["base"].append(msg.joints[i].effort)
                x.append(r)
            if topic == "/hyq/debug":
                r = dict()
                r["t"] = t.to_time()
                stance = []
                for i in range(len(msg.name)):
                    if msg.name[i] in ["stanceLF",
                                       "stanceLH",
                                       "stanceRF",
                                       "stanceRH"]:
                        stance += [msg.data[i]]
                r["stance"] = stance
                x2.append(r)
            if topic == "/hyq/des_joint_states":
                r = dict()
                r["t"] = t.to_time()
                r["pos"] = []
                for i in range(len(msg.position)):
                    r["pos"].append(msg.position[i])
                r["vel"] = []
                for i in range(len(msg.velocity)):
                    r["vel"].append(msg.velocity[i])
                y.append(r)

        return x, x2, y

    def load_pkl(self):

        self.printv("\n\n ===== Collecting Data pkl =====\n")

        with open(self.data_file , "rb") as f:
            [x, y, self.t] = pickle.load(f)

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
        self.x2_t = np.array([r["t"] for r in x2])
        if self.x2_t.shape[0] is not 0:
            self.x2_t -= x2[0]["t"]
        self.y_t = np.array([r["t"] for r in y]) - y[0]["t"]
        self.x_val = np.array([r["joint"] + r["base"] for r in x])
        self.x2_val = np.array([r["stance"] for r in x2])
        self.y_val = np.array([r["pos"] + r["vel"] for r in y])

        # Get new interpolation range
        self.data_n = max(len(x), len(x2), len(y))

        # Interpolate in new range
        t_min = min(np.min(self.x_t), np.min(self.y_t))
        t_max = max(np.max(self.x_t), np.max(self.y_t))

        x_fct = self.get_fct(self.x_t, self.x_val)
        if self.x2_t.shape[0] is not 0:
            x2_fct = self.get_fct(self.x2_t, self.x2_val)
        y_fct = self.get_fct(self.y_t, self.y_val)

        self.t = np.linspace(max(np.min(self.x_t), np.min(self.y_t)),
                             min(np.max(self.x_t), np.max(self.y_t)),
                             self.data_n)
        x_new = self.interpolate(x_fct, self.t)
        if self.x2_t.shape[0] is not 0:
            x2_new = self.interpolate(x2_fct, self.t)
        y_new = self.interpolate(y_fct, self.t)

        # Concatenate all input data
        if self.x2_t.shape[0] is not 0:
            x_new = np.hstack((x_new, x2_new))

        return x_new, y_new

    def split_data(self, x, y):

        self.printv("\n ===== Splitting Data =====\n")

        # Divide into training and testing dataset
        x_train, x_test, y_train, y_test = \
                    train_test_split(x, y, shuffle=False,
                                     test_size=(1-self.test_split))
        # Print dimensions in a table
        lin = ["HyQ Body State (in)", "Hyq joint command (out)"]
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


    ## ELEMENTARY PIPE FUNCTIONS

    def get_in_dim(self):

        # Create dummy data to get pipe length
        dummy = np.zeros(self.x_train.shape)
        y = self.in_pipe.transform(dummy)

        return y.shape

    def create_x_scaler(self):

        return esn.HyQStateScaler()
        #return MinMaxScaler((-1, 1))

    def create_y_scaler(self):

        return esn.HyQJointScaler()
        # return MinMaxScaler((-1, 1))

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

        # Output Layer
        n_out = self.n_out
        x = Dense(n_out, kernel_regularizer=l2(self.regularization))(x)
        #x = Activation('tanh')(x)

        # Compile and print network
        self.nn = Model(inputs=[state_input], outputs=x)
        self.nn.compile(loss='mae',
                   optimizer=Adam(),
                   metrics=['accuracy', 'mse', 'mae', 'mape', 'cosine'])
        if self.verbose > 1:
            self.nn.summary()

        return self.nn

    def create_nn(self):

        self.printv("\n\n ===== Creating Network =====")

        # Wrap in a regressor
        self.callbacks = [ModelCheckpoint(self.save_folder + "/best_model.h5",
                                     monitor='val_loss', verbose=self.verbose,
                                     save_best_only=True, mode='min'),
                          utils.CustomHistory(), TensorBoard()]
        if self.stop_delta is not None:
            self.callbacks += [EarlyStopping(monitor='val_loss',  mode="min",
                                       verbose=self.verbose,
                                       patience=self.stop_pat,
                                       min_delta=self.stop_delta)]
        verbose = self.verbose
        if verbose > 1:
            verbose = 1
        return KerasRegressor(build_fn=self.create_nn_fn,
                              validation_split=self.val_split,
                              batch_size=self.batch_size,
                              epochs=self.max_epochs,
                              callbacks=self.callbacks,
                              verbose=verbose)

    def create_pp(self):

        pp_pipe = []
        i = 0

        for l in self.network_layers:
            if len(l) > 1:
                tl = []
                tw = {}
                for k in l:
                    if k[0] == 'esn':
                        e = esn.SimpleESN(n_readout=self.esn_n_read,
                                          n_components=k[1],
                                          weight_scaling=k[2])
                        step_name = 'esn' + str(i)
                        tl += [(step_name, e)]
                        tw[step_name] = 1

                    if k[0] == 'esnsfa':
                        e = esn.ESNSFA(n_readout=self.esn_n_read,
                                       n_components=k[1],
                                       weight_scaling=k[2],
                                       n_res=k[3],
                                       l_min=k[4],
                                       l_max=k[5])
                        step_name = 'esnsfa' + str(i)
                        tl += [(step_name, e)]
                        tw[step_name] = 1

                    if k[0] == 'td':
                        e = esn.TimeDelay(k[1], k[2])
                        step_name = 'td' + str(i)
                        tl += [(step_name, e)]
                        tw[step_name] = 1

                    if k[0] == 'noise':
                        e = esn.GaussianNoise(stdev=k[1])
                        step_name = 'noise' + str(i)
                        tl += [(step_name, e)]
                        tw[step_name] = 1
                pp_pipe += [('fu', FeatureUnion(transformer_list=tl,
                                                transformer_weights=tw))]
            else:
                if l[0][0] == 'esn':
                    e = esn.SimpleESN(n_readout=self.esn_n_read,
                                      n_components=l[0][1],
                                      weight_scaling=l[0][2])
                    pp_pipe += [('esn' + str(i), e)]

                if l[0][0] == 'esnsfa':
                    e = esn.ESNSFA(n_readout=self.esn_n_read,
                                   n_components=l[0][1],
                                   weight_scaling=l[0][2],
                                   n_res=l[0][3],
                                   l_min=l[0][4],
                                   l_max=l[0][5])
                    pp_pipe += [('esnsfa' + str(i), e)]

                if l[0][0] == 'td':
                    e = esn.TimeDelay(l[0][1], l[0][2])
                    pp_pipe += [('td' + str(i), e)]

                if l[0][0] == 'noise':
                    e = esn.GaussianNoise(stdev=l[0][1])
                    pp_pipe += [('noise' + str(i), e)]

            i += 1

        return pp_pipe


    ## PLOT AND EVALUATION FUNCTIONS

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
        del to_save["x_train"], to_save['nn_pipe']
        del to_save["y_train"], to_save["x_test"], to_save["y_test"]
        pickle.dump(to_save, open(self.save_folder + "/network_" +
                                  str(index)+ ".pkl", "wb"), protocol=2)
        del to_save

        # Save pipe mode
        if self.nn_pipe is not None:
            pipe_to_save = copy.copy(self.nn_pipe)
            del pipe_to_save.named_steps['nn'].model
            del pipe_to_save.named_steps['nn'].build_fn
            joblib.dump(pipe_to_save, self.save_folder + "/pipe_" + \
                        str(index)+ ".pkl")

        # Save nn model
        if self.nn is not None:
            self.nn.save(self.save_folder + "/model_" + str(index)+ ".h5")


        self.printv("Files saved: " + self.save_folder + "/model_" + \
                    str(index) + ".h5, pipe_" + str(index ) + \
                     ".pkl, network_" + str(index ) + ".pkl")

    def load(self, folder, num=0, load_all=False):

        # Load this class
        verbose_rem = self.verbose
        with open(folder + "/network_" + str(num) + ".pkl",'rb') as f:
            self.__dict__ = pickle.load(f)
            self.verbose = verbose_rem
            self.reservoir = []

        # Load pipe mode
        pipe_file = folder + "/pipe_" + str(num) + ".pkl"
        if os.path.isfile(pipe_file):
            self.nn_pipe = joblib.load(pipe_file)

        # Load network
        self.nn = load_model(folder + "/model_" + str(num) + ".h5")
        if os.path.isfile(pipe_file):
            self.nn_pipe.named_steps['nn'].model = \
                load_model(folder + "/model_" + str(num) + ".h5")

        # Load dataset
        self.load_data(load_all)

    def printv(self, txt):

        if self.verbose >= 1:
            print(txt)

    def plot_histogram(self, x):

        n, bins, p = plt.hist(x.flatten(), 50, facecolor='g', alpha=0.75)

        plt.xlabel('Input values')
        plt.ylabel('Distribution')
        plt.grid(True)
        plt.show()

    def plot_fft(self):

        self.printv("\n\n ===== Computing FFT =====\n")

        # Process features
        x_ft = self.in_pipe.transform(self.x_test)
        x_ft = np.expand_dims(x_ft, axis=2)
        y_ft = self.out_pipe.transform(self.y_test)

        # Process NN
        y_pred = self.out_pipe.inverse_transform(self.nn.predict(x_ft,
                    batch_size=self.batch_size,
                    verbose=self.verbose))

        y_truth = self.y_test

        ts = self.t[1] - self.t[0]
        fs = 1.0/ts

        n = y_truth.shape[0]
        k = np.arange(n)
        T = n / fs
        frq = k / T
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

    def evaluate(self, show):

        self.printv("\n\n ===== Evaluating Test Dataset =====\n")

        # Process features

        x_ft = self.in_pipe.transform(self.x_test)
        x_ft = np.expand_dims(x_ft, axis=2)
        y_ft = self.out_pipe.transform(self.y_test)

        # self.plot_histogram(x_ft[:, :, 0])

        # Process NN
        score = self.nn.evaluate(x_ft, y_ft, verbose=2)
        y_pred_ft = self.nn.predict(x_ft, batch_size=self.batch_size,
                               verbose=self.verbose)
        y_pred = self.out_pipe.inverse_transform(y_pred_ft)

        y_truth = self.y_test
        self.printv("Test loss: " + str(score[0]))
        self.printv("Test accuracy: " + str(score[1]))
        self.test_loss = score[0]
        self.test_accuracy = score[1]

        if show:
            # Summarize history for loss
            h = self.history
            plt.plot(h['loss'])
            plt.plot(h['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Training', 'Validation'], loc='upper left')
            plt.show()

            # Plot test and predicted values
            plt.plot(y_truth[:, 0], y_pred[:, 0],
                     marker='o', linestyle='None')
            plt.xlabel("Actual value")
            plt.ylabel("Predicted value")
            plt.show()
            t = self.t[0:y_truth.shape[0]]
            plt.plot(t, y_truth[:, 0], label="real")
            plt.plot(t, y_pred[:, 0], label="predicted")
            plt.plot(t, np.abs(y_truth - y_pred)[:, 0],
                     label="MAE error")
            plt.legend()
            plt.show()

            ts = [0]
            for i in range(len(self.t) - 1):
                ts.append(self.t[i+1] - self.t[i])
            plt.plot(self.t, ts)
            plt.show()

        return y_truth, y_pred, score


    ## MAIN FUNCTIONS

    def fit_transform_ft(self, x, y, show=True):

        # Create IN pipe and transform
        self.printv("\n\n ===== Transform In Features =====\n")

        self.in_pipe = Pipeline([('xsc0', self.create_x_scaler())] + \
                                 self.create_pp())
        x_ft = self.in_pipe.fit_transform(x)
        x_ft = np.expand_dims(x_ft, axis=2)
        self.n_in = x_ft.shape[1]
        self.n_out = y.shape[1]
        self.printv("\nTotal input features size: " + str(x_ft.shape))
        self.printv("\nTotal output features size: " + str(y.shape))

        flatten_nn_struct =  utils.flatten(self.in_pipe.steps)
        if show:
            for e in flatten_nn_struct:
                 if isinstance(e, tuple):
                    if 'esnsfa' in e[0]:
                        ts = self.t[1] - self.t[0]
                        t_max = e[1].y.shape[0] * ts
                        print ts, t_max
                        t = np.arange(0, t_max, ts)
                        e[1].plot_sfa(t)


        # Create OUT pipe and transform
        self.printv("\n ===== Transform Out Features =====")
        raw_pipe = [('ysc', self.create_y_scaler())]
        self.out_pipe = Pipeline(raw_pipe)
        y_ft = self.out_pipe.fit_transform(y)

        return x_ft, y_ft

    def fit(self, x, y):

        self.nn = self.create_nn()

        self.printv("\n\n ===== Training Network =====\n")
        t_i = time.time()
        self.nn.fit(x, y)
        self.training_time = time.time() - t_i

    def train(self, show=True):

        # Get data
        self.load_data()

        # Get features and fit the network
        x_ft, y_ft = self.fit_transform_ft(self.x_train, self.y_train)
        self.fit(x_ft, y_ft)
        self.history = utils.history

        # Save Results
        self.save()

        # Show Evaluation
        self.evaluate(show)

        return self.test_loss, self.test_accuracy

    def train_step(self, x, y, ):

        if self.train_it == 0:

            # Create the network
            x_ft, y_ft = self.fit_transform_ft(x, y, show=False)
            print x_ft.shape, y_ft.shape
            self.fit(x_ft, y_ft)
            self.history = utils.history

            self.train_it += 1
            return utils.history["loss"][-1], utils.history["acc"][-1]
        else:
            x_ft = self.in_pipe.fit_transform(x)
            x_ft = np.expand_dims(x_ft, axis=2)
            y_ft = self.out_pipe.fit_transform(y)
            self.nn.fit(x_ft, y_ft)
            self.history = utils.history

            self.train_it += 1
            return utils.history["loss"][-1], utils.history["acc"][-1]
            # self.train_it += 1
            # return history[0], history[1]

    def predict(self, state):

        x_ft = self.in_pipe.transform(state)
        self.x_ft = x_ft
        x_ft = np.expand_dims(x_ft, axis=2)

        y_ft = self.nn.predict_on_batch(x_ft)
        y = self.out_pipe.inverse_transform(y_ft)
        #print x_ft.shape, y_ft.shape, state.shape, y.shape

        return y[0,:].flatten()



if __name__ == '__main__':

    if len(sys.argv) > 1:

        if sys.argv[1] == "process":
            nn = FeedForwardNN(data_file=sys.argv[2])
            nn.load_data()

        if sys.argv[1] == "test":
            nn = FeedForwardNN()
            nn.load(sys.argv[2])
            nn.evaluate(True)

        if sys.argv[1] == "fft":
            nn = FeedForwardNN()
            nn.load(sys.argv[2])
            nn.plot_fft()

        if sys.argv[1] == "train":
            folder = "data/nn_learning/"+ utils.timestamp()
            utils.mkdir(folder)
            nn = FeedForwardNN(data_file=sys.argv[2], save_folder=folder)
            nn.train(False)


        if sys.argv[1] == "sim":
            p = physics.HyQSim(view=True)
            p.start()
            p.register_node()
            nn = FeedForwardNN()
            nn.load(sys.argv[2])

            while not ros.is_shutdown():
                # Get last state
                state = p.get_hyq_state()

                # Predict next action
                if state.shape[1] == 46:
                    action = nn.predict(state)

                    # Send action
                    p.set_hyq_action(action)
                    p.send_hyq_traj()

            p.stop()