## IMPORTS
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import categorical_crossentropy, mean_absolute_error
from keras.optimizers import Adadelta
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import re
import rospy as ros
import rosbag
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import sys
import time

import reservoir
import utils

## MATPLOTLIB STYLE
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
plt.style.use('fivethirtyeight')
plt.rc('lines', linewidth= 1)
plt.rc('text', usetex=False)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)


class FeedForwardNN():

    ## ALGORITHM METAPARAMETERS

    def __init__(self, batch_size=2048, max_epochs=5000, test_split=0.7,
                 x_mem_buff_size=1, val_split=0.1, verbose=2, stop_pat=400,
                 nn_layers=[(500, 'lsm', 0.9), (2048, 'relu'),
                 (2048, 'relu')], stop_delta=0.001,
                 data_file="data/sims/simple_walk.bag",
                 save_folder="data/nn_learning/"):

        ## ALGORITHM METAPARAMETERS
        self.data_file = data_file
        self.save_folder = save_folder
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.x_mem_buff_size = x_mem_buff_size
        self.test_split = test_split
        self.val_split = val_split
        self.network_layers = nn_layers
        self.stop_delta = stop_delta
        self.stop_pat = stop_pat
        self.verbose = verbose
        self.save_folder = save_folder

        self.history = None
        self.nn = None
        self.training_time = 0
        self.loss = 0
        self.acc = 0
        self.x_scaler = None
        self.y_scaler = None

    ## DATA PROCESSING FUNCTIONS

    def load_data(self):

        if not "processed" in self.data_file:
            if ".bag" in self.data_file:
                # Load the bag file
                x, x2, y = self.load_bag()

                # Format the data
                x_new, y_new = self.format_data(x, x2, y)

                # Save data in numpy array
                with open(self.data_file.replace("bag", "pkl") , "wb") as f:
                    pickle.dump([x_new, y_new], f, protocol=2)

            if ".pkl" in self.data_file:
                x_new, y_new = self.load_pkl()

            # Add memory and LSTM
            x_new = self.add_recurs(x_new)

            # Save preprocessed data
            pn = utils.split(self.data_file, "/.")
            pn[-2] += "_processed"
            with open('/'.join(pn[:-1]) + '.' + pn[-1] , "wb") as f:
                pickle.dump([x_new, y_new], f, protocol=2)

        else:
            x_new, y_new = self.load_pkl()

        x_new, y_new = self.scale_data(x_new, y_new)
        self.split_data(x_new, y_new)

    def load_bag(self):

        self.printv("\n ===== Collecting Data Bag =====\n")

        x = []
        x2 = []
        y = []

        # Retrieve the data from the bag file
        bag = rosbag.Bag(self.data_file, 'r')
        t_init = bag.get_start_time()
        for topic, msg, t in bag.read_messages():
            if topic == "/hyq/robot_states":
                r = dict()
                r["t"] = t.to_time()
                r["base"] = [b.velocity for b in msg.base]
                r["joint"] = [b.position for b in msg.joints]
                r["joint"] += [b.velocity for b in msg.joints]
                r["joint"] += [b.effort for b in msg.joints]
                x.append(r)
            if topic == "/hyq/debug":
                r = dict()
                r["t"] = t.to_time()
                stance = []
                for i, m in enumerate(msg.name):
                    if m in ["stanceLF",
                             "stanceLH",
                             "stanceRF",
                             "stanceRH"]:
                        stance += [msg.data[i]]
                r["stance"] = stance
                x2.append(r)
            if topic == "/hyq/des_joint_states":
                r = dict()
                r["t"] = t.to_time()
                r["pos"] = list(msg.position)
                r["vel"] = list(msg.velocity)
                y.append(r)

        return x, x2, y

    def load_pkl(self):

        self.printv("\n ===== Collecting Data pkl =====\n")

        with open(self.data_file , "rb") as f:
            [x, y] = pickle.load(f)

        return x, y

    def get_fct(self, x, y):

        y_col = y.shape[1]
        fct_list = []

        for i in range(y_col):
            x_new, j = np.unique(x, return_index=True)
            y_new = np.array(y[:, i])[j]
            fct_list.append(interp1d(x_new, y_new, assume_sorted=False))

        return fct_list

    def interpolate(self, fct, x):

        y = []
        for f in fct:
            y.append(f(x))

        return np.array(y).T

    def format_data(self, x, x2, y):

        self.printv("\n ===== Formating Data =====\n")

        # Get values
        x_t = np.array([r["t"] for r in x])
        x2_t = np.array([r["t"] for r in x2])
        y_t = np.array([r["t"] for r in y])
        x_val = np.array([r["joint"] + r["base"] for r in x])
        x2_val = np.array([r["stance"] for r in x2])
        y_val = np.array([r["pos"] + r["vel"] for r in y])

        # Get new interpolation range
        self.data_n = max(len(x), len(x2), len(y))

        # Interpolate in new range
        x_fct = self.get_fct(x_t, x_val)
        x2_fct = self.get_fct(x2_t, x2_val)
        y_fct = self.get_fct(y_t, y_val)
        t_new = np.linspace(max(np.min(x_t), np.min(y_t)),
                            min(np.max(x_t), np.max(y_t)), self.data_n)
        x_new = self.interpolate(x_fct, t_new)
        x2_new = self.interpolate(x2_fct, t_new)
        y_new = self.interpolate(y_fct, t_new)

        # Concatenate all input data
        x_new = np.hstack((x_new, x2_new))

        return x_new, y_new

    def scale_data(self, x, y):

        self.printv("\n\n ===== Scaling Data =====\n")

        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.x_scaler.fit(x)
        self.y_scaler.fit(y)

        x_old_min = min(x)
        x_old_max = max(x)
        y_old_min = min(y)
        y_old_max = max(y)
        x = self.x_scaler.transform(x)
        y = self.y_scaler.transform(y)

        print "Input previous range: [" + str(x_old_min) + \
              ", " + str(x_old_max) + "] and new range: [" + \
              str(min(x)) + ", " + str(max(x)) + "]"
        print "Output previous range: [" + str(y_old_min) + \
              ", " + str(y_old_max) + "] and new range: [" + \
              str(min(y)) + ", " + str(max(y)) + "]\n"

        return x, y

    def split_data(self, x, y):

        # Divide into training and testing dataset
        x_train = x[0:int(x.shape[0]*self.test_split), :]
        y_train = y[0:int(y.shape[0]*self.test_split), :]
        x_test = x[int(x.shape[0]*self.test_split):, :]
        y_test = y[int(y.shape[0]*self.test_split):, :]

        # Expend dims for the NN
        x_train = np.expand_dims(x_train, axis=2)
        x_test = np.expand_dims(x_test, axis=2)

        # Print dimensions in a table
        lin = ["HyQ Body State (in)", "Hyq joint command (out)"]
        col = ["Training", "Validation", "Testing"]
        dat = [[(int(x_train.shape[0]*(1-self.val_split)),
                 x_train.shape[1], x_train.shape[2]),
                (int(y_train.shape[0]*(1-self.val_split)), y_train.shape[1])],
               [(int(x_train.shape[0]*self.val_split),
                 x_train.shape[1], x_train.shape[2]),
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

    def add_recurs(self, x):

        # Add memory buffer on the input data
        x_shift = copy.copy(x)
        x_buff = copy.copy(x)
        for i in range(1, self.x_mem_buff_size):
            x_shift = np.vstack((np.zeros((i, x.shape[1])),
                                 x[:-i, :]))
            x_buff = np.hstack((x_buff, x_shift))
        x = copy.copy(x_buff)
        del x_buff, x_shift

        # Shift prediction of one time step
        x = np.vstack((np.zeros((1, x.shape[1])), x[:-1, :]))

        # Add a LSM
        for l in self.network_layers:
            if l[1] == 'lsm':
                r = reservoir.ReservoirNet(n_in=x.shape[1])
                x = r.run(x)

        return x

    def unscale_data(self, x, y):

        x = self.x_scaler.inverse_transform(x)
        y = self.y_scaler.inverse_transform(y)

        return x, y

    ## KERAS NETWORK FUNCTIONS

    def create_nn(self):

        self.printv("\n\n\n ===== Creating Network =====")

        # Input Layer
        n_in = self.x_train.shape[1]
        state_input = Input(shape=(n_in, 1, ),
                            name='robot_state')
        x = Flatten()(state_input)

        # Network layers
        for l in self.network_layers:
            if l[1] != 'lsm':
                x = Dense(l[0])(x)
                x = Activation(l[1])(x)

        # Output Layer
        n_out = self.y_train.shape[1]
        x = Dense(n_out)(x)
        x = Activation('tanh')(x)

        # Compile and print network
        self.nn = Model(inputs=[state_input], outputs=x)
        self.nn.compile(loss='mse',
                   optimizer=Adadelta(),
                   metrics=['accuracy', 'mse', 'mae', 'mape', 'cosine'])
        if self.verbose > 1:
            self.nn.summary()

    ## PLOT AND EVALUATION FUNCTIONS

    def save(self):

        self.printv("\n\n ===== Saving =====")

        f_in_folder = [f for f in listdir(self.save_folder)
                       if isfile(join(self.save_folder, f))]

        print f_in_folder

        # Save training data
        to_save = copy.copy(self.__dict__)
        del to_save["nn"], to_save["history"].model, to_save["x_train"]
        del to_save["y_train"], to_save["x_test"], to_save["y_test"]
        pickle.dump(to_save, open(self.save_folder + "/network.pkl", "wb"),
                    protocol=2)
        del to_save

        # Save model
        if self.nn is not None:
            self.nn.save(self.save_folder + "/model.h5")

    def load(self, folder):

        # Load this class
        with open(folder + "/network.pkl",'rb') as f:
            self.__dict__ = pickle.load(f)

        # Load network
        self.nn = load_model(folder + "/model.h5")

        # Load dataset
        self.load_data()

    def printv(self, txt):

        if self.verbose > 1:
            print(txt)

    def plot_histogram(self, x):

        n, bins, p = plt.hist(x, 50, facecolor='g', alpha=0.75)

        plt.xlabel('Input values')
        plt.ylabel('Distribution')
        plt.grid(True)
        plt.show()

    def evaluate(self, show):

        self.printv("\n\n ===== Evaluating Test Dataset =====\n")

        score = self.nn.evaluate(self.x_test, self.y_test, verbose=2)
        y_pred = self.y_scaler.inverse_transform(self.nn.predict(self.x_test,
                                                 batch_size=self.batch_size,
                                                 verbose=self.verbose))
        y_truth = self.y_scaler.inverse_transform(self.y_test)
        self.printv("Test loss: " + str(score[0]))
        self.printv("Test accuracy: " + str(score[1]))
        self.test_loss = score[0]
        self.test_accuracy = score[1]

        if show:
            # Summarize history for accuracy
            plt.plot(self.history.history['acc'])
            plt.plot(self.history.history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

            # Summarize history for loss
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

            # Plot test and predicted values
            plt.plot(y_truth[:, 0], label="real")
            plt.plot(y_pred[:, 0], label="predicted")
            plt.plot(np.abs(y_truth - y_pred)[:, 0], label="MAE error")
            plt.legend()
            plt.show()

        return y_truth, y_pred

    ## MAIN FUNCTION

    def train(self):

        self.printv("\n\n ===== Training Network =====\n")

        t_i = time.time()
        callbacks = [ModelCheckpoint(self.save_folder + "/best_model.h5",
                                     monitor='val_acc', verbose=self.verbose,
                                     save_best_only=True, mode='max')]
        if self.stop_delta is not None:
            callbacks += [EarlyStopping(monitor='val_acc',  mode="max",
                                       verbose=self.verbose,
                                       patience=self.stop_pat,
                                       min_delta=self.stop_delta)]

        self.history = self.nn.fit(self.x_train, self.y_train,
                         validation_split=self.val_split,
                         batch_size=self.batch_size,
                         epochs=self.max_epochs,
                         callbacks=callbacks,
                         verbose=self.verbose)

        self.training_time = time.time() - t_i

    def run(self, show=True):

        # Get data
        self.load_data()

        # Create network
        self.create_nn()

        # Train Network
        self.train()

        # Save Results
        self.save()

        # Show Evaluation
        self.evaluate(show)

        return self.test_loss, self.test_accuracy


if __name__ == '__main__':

    if len(sys.argv) > 1:

        if sys.argv[1] == "process_data":
            nn = FeedForwardNN(data_file=sys.argv[2])
            nn.load_data()

        if sys.argv[1] == "test":
            nn = FeedForwardNN()
            nn.load(sys.argv[2])
            nn.evaluate(False)

        if sys.argv[1] == "train":
            folder = "data/nn_learning/"+ utils.timestamp()
            utils.mkdir(folder)
            nn = FeedForwardNN(data_file=sys.argv[2], save_folder=folder)
            nn.run(False)