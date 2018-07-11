
from contextlib import contextmanager
from io import BytesIO as StringIO
import os
import pickle
import tensorflow as tf
import time
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import network
import utils

SAVE_FOLDER = "data/off_learning_lsm/"
DATA = "data/sims/rough_walk.pkl"


def create_nn_pool():

    layer_sizes =  [1024, 4096]
    layer_num = 3
    activation = ['relu', 'tanh']
    lsm_size = [0, 100, 1000]
    lsm_spec_rad = [0.7, 0.9]

    nn_layers = []

    for lsm_s in lsm_size:
        for lsm_r in lsm_spec_rad:
            if lsm_s == 0:
                lsm_nn = []
            else:
                lsm_nn = [(lsm_s, 'lsm', lsm_r)]
            for a in activation:
                l_array = tuple()
                for l in range(layer_num):
                    l_array += (layer_sizes,)
                layers = utils.cartesian(l_array)
                for nn in layers:
                    nn_layers.append(lsm_nn + [(n, a) for n in nn])

    return nn_layers


if __name__ == '__main__':

    # Create a results file
    folder = SAVE_FOLDER + utils.timestamp()
    utils.mkdir(folder)

    # Create pool of network architectures and browse
    nn_layers = create_nn_pool()
    for i, e in enumerate(nn_layers):

        # Create new experiment
        print("\n ===== Experiment "+ str(i+1) + "/" +
              str(len(nn_layers)) + " ===== ")
        print("NN Architecture: " + str(e))
        nn = network.FeedForwardNN(nn_layers=e, data_file=DATA,
                                   save_folder=folder, verbose=1)

        # Start and time experiment
        t_i = time.time()
        l, a = nn.run(show=False)
        t = time.time() - t_i

        # Print main metrics
        print("\nTest Accuracy: {:0.5f}".format(a))
        print("Test Loss: {:0.5f}".format(l))
        print("Training Time: {:0.2f}".format(t))
        print("Training Epochs: " + str(len(nn.history.epoch)))