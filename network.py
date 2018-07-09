## IMPORTS
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy, mean_absolute_error
from keras.optimizers import Adadelta
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
import matplotlib.pyplot as plt
import numpy as np
import pickle
import rospy as ros
import rosbag
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler


## MATPLOTLIB STYLE
plt.style.use('fivethirtyeight')
plt.rc('lines', linewidth= 1)
plt.rc('text', usetex=False)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)


class FeedForwardNN():

    ## ALGORITHM METAPARAMETERS

    def __init__(self, batch_size=64, epochs=5000, test_split=0.7,
                 val_split=0.1, nn_layers=[(1024, 'relu')],
                 stop_delta=0.00001, stop_pat=50, verbose=2,
                 file="data/sims/simple_walk.bag"):

        ## ALGORITHM METAPARAMETERS
        self.file = file
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_split = test_split
        self.val_split = val_split
        self.network_layers = nn_layers
        self.stop_delta = stop_delta
        self.stop_pat = stop_pat
        self.verbose = verbose


        self.x_scaler = None
        self.y_scaler = None

    ## DATA PROCESSING FUNCTIONS

    def get_data(self, file):

        if ".pkl" in file:
            return self.get_pkl(file)
        elif ".bag" in file:
            return self.get_bag(file)

    def get_bag(self, file):

        x = []
        x2 = []
        y = []

        # Retrieve the data from the bag file
        bag = rosbag.Bag(file, 'r')
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

        # Save usefull data to a pickle file to restore more quickly
        with open(file.replace("bag", "pkl") , "wb") as f:
            pickle.dump([x, x2, y], f, protocol=2)

        return x, x2, y

    def get_pkl(self, file):

        with open(file , "rb") as f:
            [x, x2, y] = pickle.load(f)

        return x, x2, y

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

    def format_dataset(self, x, x2, y):

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
        print x_new.shape, x2_new.shape
        x_new = np.hstack((x_new, x2_new))
        print x_new.shape

        # Shift prediction of one time step
        x_new = np.vstack((np.zeros((1, x_new.shape[1])), x_new[1:, :]))

        # Divide into training and testing dataset
        x_train = x_new[0:int(x_new.shape[0]*self.test_split), :]
        y_train = y_new[0:int(y_new.shape[0]*self.test_split), :]
        x_test = x_new[int(x_new.shape[0]*self.test_split):, :]
        y_test = y_new[int(y_new.shape[0]*self.test_split):, :]

        # Scale
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.x_scaler.fit(x_train)
        x_train = self.x_scaler.transform(x_train)
        x_test = self.x_scaler.transform(x_test)
        self.y_scaler.fit(y_train)
        y_train = self.y_scaler.transform(y_train)
        y_test = self.y_scaler.transform(y_test)

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

        return x_train, y_train, x_test, y_test

    def unscale(self, x, y):

        global x_scaler, y_scaler
        x = x_scaler.inverse_transform(x)
        y = y_scaler.inverse_transform(y)

        return x, y

    ## KERAS NETWORK FUNCTIONS

    def create_nn(self, n_in, n_out):

        # Input Layer
        state_input = Input(shape=(n_in, 1, ),
                            name='robot_state')
        x = Flatten()(state_input)

        # Network layers
        for l in self.network_layers:
            x = Dense(l[0])(x)
            x = Activation(l[1])(x)

        # Output Layer
        x = Dense(n_out)(x)
        x = Activation('tanh')(x)

        # Compile and print network
        nn = Model(inputs=[state_input], outputs=x)
        nn.compile(loss='mse',
                   optimizer=Adadelta(),
                   metrics=['accuracy', 'mse', 'mae', 'mape', 'cosine'])
        if self.verbose > 1:
            nn.summary()
        return nn

    ## PLOT AND EVALUATION FUNCTIONS

    def printv(self, txt):

        if self.verbose > 1:
            print(txt)

    def plot_hist(self, x):

        n, bins, p = plt.hist(x, 50, facecolor='g', alpha=0.75)

        plt.xlabel('Input values')
        plt.ylabel('Distribution')
        plt.grid(True)
        plt.show()

    def plot_evaluation(self, hist, x_test, y_test, show):

        score = self.nn.evaluate(x_test, y_test, verbose=2)
        y_pred = self.nn.predict(x_test, batch_size=self.batch_size,
                                 verbose=self.verbose)
        self.printv("Test loss: " + str(score[0]))
        self.printv("Test accuracy: " + str(score[1]))

        if show:
            # Summarize history for accuracy
            plt.plot(hist.history['acc'])
            plt.plot(hist.history['val_acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

            # Summarize history for loss
            plt.plot(hist.history['loss'])
            plt.plot(hist.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

            # Plot test and predicted values
            plt.plot(y_test[:, 0], label="real")
            plt.plot(y_pred[:, 0], label="predicted")
            plt.plot(np.abs(y_test - y_pred)[:, 0], label="MAE error"),
            plt.legend()
            plt.show()

        return score[0], score[1]

    ## MAIN FUNCTION

    def run(self, show=True):

        # Get data
        self.printv("\n ===== Collecting Data from Rosbag =====")
        x, x2, y = self.get_data(self.file)
        x_train, y_train, x_test, y_test = self.format_dataset(x, x2, y)

        # Create network
        self.printv("\n\n\n ===== Creating Network =====")
        self.nn = self.create_nn(x_train.shape[1], y_train.shape[1])

        # Train Network
        self.printv("\n\n ===== Training Network =====\n")
        callbacks = [EarlyStopping(monitor='val_loss',  mode="min",
                                   verbose=self.verbose, patience=self.stop_pat,
                                   min_delta=self.stop_delta)]
        history = self.nn.fit(x_train, y_train,
                         validation_split=self.val_split,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         callbacks=callbacks,
                         verbose=self.verbose)

        # Show Evaluation
        self.printv("\n\n ===== Showing Results =====")
        loss, acc = self.plot_evaluation(history, x_test, y_test, show)

        return loss, acc, history



if __name__ == '__main__':

    nn_layers = [(2048, 'relu'), (2048, 'relu')]
    nn = FeedForwardNN(nn_layers=nn_layers, file="data/sims/rough_walk.pkl")
    nn.run(True)