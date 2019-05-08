from contextlib import contextmanager
import copy
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import psutil
import random
from sklearn.pipeline import FeatureUnion
import signal
import sys
import tempfile
import time
from tqdm import tqdm_notebook, tqdm
import warnings


def timestamp():
    """ Return a string stamp with current date and time """

    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def mkdir(path):
    """ Create a directory if it does not exist yet """

    if not os.path.exists(path):
        os.makedirs(path)


def cleanup():

    print('\n\n -- Quitting and killing all children processes! -- \n')
    process = psutil.Process()
    children = process.children(recursive=True)
    time.sleep(0.2)
    for p in children:
        p.kill()
    process.kill()


def kill_children():

    print('\n\n -- Killing all children processes! -- \n')
    process = psutil.Process()
    children = process.children(recursive=True)
    time.sleep(0.2)
    for p in children:
        p.kill()


def save_on_top(newdata, filename):
    """ Append data to a file that is already saved """

    if os.path.exists(filename):
        data = pickle.load(open(filename, "rb"))
        final = list(copy.copy(data)) + [copy.copy(newdata)]
    else:
        final = [copy.copy(newdata)]

    pickle.dump(final, open(filename, "wb"), protocol=2)


def split(str, delim=" "):
    index = 0
    string = ""
    array = []
    while index < len(str):
        if str[index] not in delim:
            string += str[index]
        else:
            if string:
                array.append(string)
                string = ""
        index += 1
    if string: array.append(string)
    return array


def mse(arr1, arr2):
    """ Compute MSE between two arrays (1D) """

    assert arr1.shape == arr2.shape, "Mean Square Error can only be computed on matrices with same size"
    a, b = np.matrix(arr2).shape
    return np.sum(np.square(arr2 - arr1)) / float(a * b)


def nrmse(arr1, arr2):
    """ Compute NRMSE between two arrays (1D) """

    # Center signals around 0?
    rmse = np.sqrt(mse(arr1, arr2))
    max_val = max(np.max(arr1), np.max(arr2))
    min_val = min(np.min(arr1), np.min(arr2))

    return rmse / (max_val - min_val)

def signaltonoise(a, axis=0, ddof=0):
    """ Compute a Signal to Noise Ratio using only mean and std deviation
    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


class Timeout():
    """Timeout class using ALARM signal."""

    class Timeout(Exception):
        pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()