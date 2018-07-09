
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


def create_nn_pool():

    batch_size = [1024]
    layer_num = range(1,8)
    layer_sizes =  [512, 4096]
    activation = 'relu'

    experiment = []

    for b in batch_size:
        for l in layer_num:
            l_array = tuple()
            for n in range(l):
                l_array += (layer_sizes,)
            layers = utils.cartesian(l_array)
            for nn in layers:
                experiment.append({"batch_size": b,
                                   "nn": [(n, activation) for n in nn]})
    return experiment


if __name__ == '__main__':

    # Create a results file
    folder = "data/results/off_learning/" + utils.timestamp()
    utils.mkdir(folder)
    filename = folder + "/results.pkl"

    # Create pool of network architectures and browse
    exp_params = create_nn_pool()
    for i, e in enumerate(exp_params):

        # Create new experiment
        print("\n ===== Experiment "+ str(i+1) + "/" +
              str(len(exp_params)) + " ===== ")
        print("Batch Size: " + str(e["batch_size"]))
        print("NN Architecture: " + str(e["nn"]))
        nn = network.FeedForwardNN(batch_size=e["batch_size"],
                                   nn_layers=e["nn"], verbose=2)

        # Start and time experiment
        t_i = time.time()
        with StringIO() as f:
            with utils.RedirectStdStreams(stdout=f, stderr=f):
                l, a, h = nn.run(show=False)
                log = f.getvalue()
        t = time.time() - t_i

        # Print main metrics
        print("Test Accuracy: {:0.5f}".format(a))
        print("Test Loss: {:0.5f}".format(l))
        print("Training Time: {:0.2f}".format(t))
        print("Training Epochs: " + str(len(h.epoch)))

        # Save results
        del h.model
        utils.save_on_top({"params": e, "test_loss": l, "test_acc": a,
                        "train_time": t, "history": h }, filename)