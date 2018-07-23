from contextlib import contextmanager
import copy
from cStringIO import StringIO
import keras
import numpy as np
import os
import pickle
import psutil
from sklearn.pipeline import FeatureUnion
import sys
import tempfile
import time

history = {}
epoch = []

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


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


class RedirectStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

class CustomHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):

        epoch = []
        history = {}

    def on_epoch_end(self, e, logs=None):

        logs = logs or {}
        epoch.append(e)
        for k, v in logs.items():
            history.setdefault(k, []).append(v)

def flatten(lst):
    new_lst = []
    flatten_helper(lst, new_lst)
    return new_lst

def flatten_helper(lst, new_lst):
    for element in lst:
        if isinstance(element, list):
            flatten_helper(element, new_lst)
        if isinstance(element, tuple):
            flatten_helper(element, new_lst)
        if isinstance(element, FeatureUnion):
            flatten_helper(element.transformer_list, new_lst)
        else:
            new_lst.append(element)

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout