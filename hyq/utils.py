from contextlib import contextmanager
import copy
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import psutil
from sklearn.pipeline import FeatureUnion
import signal
import sys
import tempfile
from tensorflow.python.client import device_lib
import time
from tqdm import tqdm_notebook, tqdm
import warnings


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

 
def main():
    # Run block of code with timeouts
    try:
        with Timeout(3):
            print test_request("Request 1")
        with Timeout(1):
            print test_request("Request 2")
    except Timeout.Timeout:
        print "Timeout"

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


def identity(x):

    return x


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


class NetworkSave(keras.callbacks.Callback):

    def __init__(self, data, monitor='val_loss', verbose=0,
                 save_best_only=False, mode='auto', period=1,
                 max_epochs=100):

        super(NetworkSave, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.max_epochs = max_epochs
        [self.folder, self.index, self.class_to_save] = data
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def to_save(self):

        # Get ESN data
        to_save2 = {}
        if 'esn' in self.class_to_save.keys():
            to_save2 = copy.copy(self.class_to_save['esn'].__dict__)
            del to_save2["keras_model"]

        # Save training data
        to_save = copy.copy(self.class_to_save)
        if 'esn' in to_save.keys():
            del to_save["esn"]
        if 'readout' in to_save.keys():
            del to_save["readout"]
        if 'callbacks' in to_save.keys():
            new_cb = []
            for i, c in enumerate(to_save["callbacks"]):
                n = c.__class__.__name__
                if n not in  ["NetworkSave", "TensorBoard", "PlotJupyter"]:
                    new_c = copy.copy(c)
                    del new_c.model
                    new_cb.append(new_c)
            del to_save["callbacks"]
            to_save["callbacks"] = new_cb

        if 'x_train' in to_save.keys():
            del to_save["x_train"], to_save["y_train"]
            del to_save["x_test"], to_save["y_test"]

        return to_save, to_save2

    def on_epoch_end(self, epoch, logs=None):

        # Forbid to kill during the saving
        s = signal.signal(signal.SIGINT, signal.SIG_IGN)

        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print("\nEpoch {:03d}: ".format(epoch + 1) + \
                                  self.monitor + " improved" + \
                                  " from {:0.5f} ".format(self.best) + \
                                  " to {:0.5f},".format(current) + \
                                  " saving model to " + self.folder)
                        self.best = current
                        self.class_to_save["epoch"] = epoch + 1
                        self.class_to_save["history"] = history

                        nn, esn = self.to_save()
                        with open(self.folder + "/esn_" + str(self.index) + ".pkl", "wb") as f:
                            pickle.dump(esn, f, protocol=2)
                        with open(self.folder + "/nn_" + str(self.index) + ".pkl", "wb") as f:
                            pickle.dump(nn, f, protocol=2)
                        self.model.save(self.folder + "/readout_" + str(self.index)+ ".h5")
                        del nn, esn

                    else:
                        if self.verbose > 0:
                            print('\nEpoch %03d: %s did not improve from %0.5f' % (epoch + 1, self.monitor, self.best))

            else:
                if self.verbose > 0:
                    print('\nEpoch %03d: saving model to %s' % (epoch + 1, self.folder))

                self.class_to_save["epoch"] = epoch + 1
                self.class_to_save["history"] = history

                nn, esn = self.to_save()
                with open(self.folder + "/esn_" + str(self.index) + ".pkl", "wb") as f:
                    pickle.dump(esn, f, protocol=2)
                with open(self.folder + "/nn_" + str(self.index) + ".pkl", "wb") as f:
                    pickle.dump(nn, f, protocol=2)
                self.model.save(self.folder + "/readout_" + str(self.index) + ".h5")
                del nn, esn

        # Know we can interupt again
        signal.signal(signal.SIGINT, s)


class PlotJupyter(keras.callbacks.Callback):

    def __init__(self, x, y, network=None, plot=False, max_epochs=100, verbose=-1):

        self.x = x
        self.y_tgt = y
        self.nn = network
        self.plot = plot
        self.max_epochs = max_epochs
        self.verbose = verbose
        if self.verbose == -1 or self.verbose == -2:
            self.t = tqdm_notebook(total=self.max_epochs)
        else:
            self.t = tqdm(total=self.max_epochs)

        self.i = 0
        self.it = None
        self.losses = None
        self.val_losses = None
        self.y_pred = None
        self.fig = None
        self.ax1 = None
        self.ax2 = None

    def on_train_begin(self, logs={}):

        self.i = 0
        self.it = []
        self.losses = []
        self.val_losses = []
        self.y_pred = np.zeros(self.y_tgt.shape)

        #self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=50)

    def on_epoch_end(self, epoch, logs={}):

        if self.verbose > -3:
            self.t.set_postfix(best_val_loss="{:.4f}".format(logs.get("val_loss")))

        self.t.update()
        self.it.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        if self.plot:

            if epoch == 0:
                if self.verbose == -3:
                    plt.ion()
                    self.fig = plt.figure(figsize=(12, 5), dpi=100)
                else:
                    self.fig = plt.figure(figsize=(12, 5), dpi=50)
                self.ax1 = self.fig.add_subplot(121)
                self.ax2 = self.fig.add_subplot(122)
                plt.show()

            if epoch % 5 == 0:

                new_esn = copy.copy(self.nn.esn)
                new_esn.keras_model = self.model

                self.y_pred = new_esn.predict(self.x)
                self.ax2.clear()
                self.ax2.plot(self.y_pred[:, 1], label='Test Prediction')
                self.ax2.plot(self.y_tgt[:, 1], label='Test Target')
                self.ax2.set_ylabel('HAA Position [rad]')
                self.ax2.set_xlabel('Time [s]')
                self.ax2.legend(loc="upper center")

                self.ax1.clear()
                self.ax1.plot(self.it, self.losses, label='Training')
                self.ax1.plot(self.it, self.val_losses, label='Validation')
                self.ax1.set_ylabel('Loss [MAE]')
                self.ax1.set_xlabel('Epoch [#]')
                self.ax1.legend(loc="upper right")

                plt.pause(0.001)
                plt.show()


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