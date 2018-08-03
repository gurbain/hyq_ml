from hyperopt import fmin
from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp
import numpy as np
import pickle
import sys

import network
import utils

MAX_EVALS = 1000
SAVE_FOLDER = "data/hypopt/"
DATA = "data/sims/rough_walk.pkl"


def objective(p):

    nn_layers = [(400, 'lsm', int(p['lsm_sr']))]

    for i in range(2):
        nn_layers += [(2048, "relu")]

    nn = network.FeedForwardNN(max_epochs=200, nn_layers=nn_layers,
                               batch_size=1024, verbose=1,
                               lsm_n_read=46,
                               data_file=DATA, save_folder=folder)
    nn.train(show=False)
    loss = np.min(nn.history['val_loss'])

    return {'loss': loss, 'params': p, 'status': STATUS_OK}


def run_hyperoptim(space, folder):

    trials_step = 1
    curr_max = 1

    # Load or create the trial object
    try:
        trials = pickle.load(open(folder + "/hyperopt.pkl", "rb"))
        curr_max = len(trials.trials) + trials_step
        print("\n ===== Experiment "+ str(len(trials.trials)) + "/" +
              str(MAX_EVALS) + " ===== \n")
    except:
        print("\n ===== Experiment 0/" + str(MAX_EVALS) + " ===== \n")
        trials = Trials()

    # Perform optimization
    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=curr_max, trials=trials)
    print "\nLast loss: " + str(trials.results[-1]['loss'])
    print "Last params: " + str(trials.results[-1]['params'])
    print "\nCurrent best loss: " + \
          str(min([r["loss"] for r in trials.results]))
    print "Current best params: " + str(best)


    # save the trials object
    with open(folder + "/hyperopt.pkl", "wb") as f:
        pickle.dump(trials, f)

    return curr_max


if __name__ == '__main__':

    # Create hyper parameter search space
    space = {
        #"s_lsm": hp.quniform('lsm_sr', 0, 1000, 50),
        #"lsm_sr": hp.uniform('lsm_sr', 0.1, 1.7),
        "n_l": hp.quniform('n_l', 1, 4, 1),
        "s_l": hp.quniform('s_l', 200, 8000, 200),
        # "act": hp.choice('act', ['relu', 'tanh'])
    }

    # Result save folder
    global folder
    if len(sys.argv) > 1:
        if sys.argv[1] == "continue":
            folder = sys.argv[2]
    else:
        folder = SAVE_FOLDER + utils.timestamp()
        utils.mkdir(folder)

    # Optimize
    it = 0
    while it < MAX_EVALS:
        it = run_hyperoptim(space, folder)