from hyperopt import fmin
from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp
import numpy as np
import pickle

import network
import utils

MAX_EVALS = 1000
SAVE_FOLDER = "data/hyperparameter_optim/"
DATA = "data/sims/rough_walk.pkl"


def objective(p):

    nn_layers = []
    if p['s_lsm'] != 0:
        nn_layers += [(int(p['s_lsm']), 'lsm', p['lsm_sr'])]

    for i in range(int(p["n_l"])):
        nn_layers += [(int(p['s_l']), p['act'])]

    nn = network.FeedForwardNN(max_epochs=100, nn_layers=nn_layers,
                               batch_size=512, verbose=1,
                               data_file=DATA, save_folder=folder)
    nn.run(show=False)
    loss = np.min(nn.history.history['val_loss'])

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
        print("\n ===== Experiment 0/" + str(MAX_EVALS) + " ===== ")
        trials = Trials()

    # Perform optimization
    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=curr_max, trials=trials)
    print "Current best loss: " + \
          str(min([r["loss"] for r in trials.results]))
    print "Current best params: " + str(best)

    # save the trials object
    with open(folder + "/hyperopt.pkl", "wb") as f:
        pickle.dump(trials, f)

    return curr_max


if __name__ == '__main__':

    # Create hyper parameter search space
    space = {
        "s_lsm": hp.choice('s_lsm', [0, 100, 200, 500, 1000]),
        "lsm_sr": hp.uniform('lsm_sr', 0.5, 1),
        "n_l": hp.quniform('n_l', 1, 4, 1),
        "s_l": hp.qloguniform('s_l', 2, 7, 1),
        "act": hp.choice('act', ['relu', 'tanh'])
    }

    # Result save folder
    global folder
    folder = SAVE_FOLDER + utils.timestamp()
    utils.mkdir(folder)

    # Optimize
    it = 0
    while it < MAX_EVALS:
        it = run_hyperoptim(space, folder)