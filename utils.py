from contextlib import contextmanager
import copy
from cStringIO import StringIO
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
import time
from tqdm import tqdm_notebook
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

        if self.verbose == -1:
            self.t = tqdm_notebook(total=self.max_epochs)


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

        # Save training data
        to_save = copy.copy(self.class_to_save)
        if 'nn' in to_save.keys():
            del to_save["nn"]
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

        return to_save

    def on_epoch_end(self, epoch, logs=None):

        # Forbid to kill during the saving
        s = signal.signal(signal.SIGINT, signal.SIG_IGN)

        # If verbose mode is bar progress (-1), update it
        if self.verbose == -1:
            self.t.set_postfix(best_val_loss="{:.4f}".format(self.best))
            self.t.update()

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
                        with open(self.folder + "/network_" + \
                                  str(self.index) + ".pkl", "wb") as f:
                            pickle.dump(self.to_save(), f, protocol=2)

                        self.model.save(self.folder + "/model_" + \
                                        str(self.index)+ ".h5")
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %03d: %s did not improve from %0.5f' % (epoch + 1, self.monitor, self.best))

            else:
                if self.verbose > 0:
                    print('\nEpoch %03d: saving model to %s' % (epoch + 1, self.folder))

                self.class_to_save["epoch"] = epoch + 1
                self.class_to_save["history"] = history

                with open(self.folder + "/network_" + \
                          str(self.index) + ".pkl", "wb") as f:
                    pickle.dump(self.to_save(), f, protocol=2)

                self.model.save(self.folder + "/model_" + \
                                str(self.index)+ ".h5")

        # Know we can interupt again
        signal.signal(signal.SIGINT, s)


class PlotJupyter(keras.callbacks.Callback):

    def __init__(self, x, y, network):

        self.x = x
        self.y_tgt = y
        self.nn = network

    def on_train_begin(self, logs={}):
        self.i = 0
        self.it = []
        self.losses = []
        self.val_losses = []
        self.y_pred = np.zeros(self.y_tgt.shape)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=50)
        plt.plot()

    def on_epoch_end(self, epoch, logs={}):

        self.it.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        self.ax1.clear()
        self.ax1.plot(self.it, self.losses, label='Training')
        self.ax1.plot(self.it, self.val_losses, label='Validation')
        self.ax1.set_ylabel('Loss [MAE]')
        self.ax1.set_xlabel('Epoch [#]')
        self.ax1.legend(loc="upper right")

        if epoch%5 == 0:

            if epoch == 0:
                self.y_pred = self.nn.predict(self.x)
            else:
                x_ft = self.nn.predict_in_pipe.transform(self.x)
                x_ft = np.expand_dims(x_ft, axis=2)
                y_ft = self.model.predict_on_batch(x_ft)
                self.y_pred = self.nn.predict_out_pipe.inverse_transform(y_ft)

            self.ax2.clear()
            self.ax2.plot(self.y_pred[:, 0], label='Test Prediction')
            self.ax2.plot(self.y_tgt[:, 0], label='Test Target')
            self.ax2.set_ylabel('HAA Position [rad]')
            self.ax2.set_xlabel('Time [s]')
            self.ax2.legend(loc="upper center")

        self.fig.canvas.draw()



