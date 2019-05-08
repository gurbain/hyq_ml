import os
import numpy as np
import pickle
import sys

import matplotlib
import matplotlib.colors as cols
import matplotlib.pyplot as plt

from hyq import analysis

plt.style.use('fivethirtyeight')
#plt.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.unicode'] = True
#plt.rc('font', size=12, family='serif', serif='Times')
plt.rc('text', usetex=False)
plt.rc('font', size=10)
plt.rc('axes', facecolor='white', edgecolor="white", labelcolor="#606060")
plt.rc('xtick', color="white")
plt.rc('ytick', color="white")
plt.rc('figure', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=False)


FOLDER = "/home/gurbain/docker_sim/experiments/locus"
IMG = "/home/gabs48/locus.png"

def sort_data(data):

    x_list = [eval(d["physics_init_impedance"])[2] for d in data]
    x = sorted(list(set(x_list)))

    f1 = [None] * len(x)
    f2 = [None] * len(x)
    for d in data:
        x_index = x.index(int(eval(d["physics_init_impedance"])[2]))
        f1[x_index] = d["f1"][-1500:]
        f2[x_index] = d["f2"][-1500:]

    return x, np.array(f1), np.array(f2)


def scatter_foot_trace(foot_trace, x_index, label=None):

    x_space = 0.2
    plt.scatter(np.array(foot_trace[:, 0] + x_index * x_space),
                np.array(foot_trace[:, 2]), s=3)

    plt.plot(np.array(foot_trace[:, 0] + x_index * x_space),
            np.array(foot_trace[:, 2]), linewidth=0.5, label=label)


def plot_locus(data):

    x, f1, f2 = sort_data(data)
    plt.figure(figsize=(18, 4), dpi=80)

    for x_c in [100, 250, 400, 550, 700, 850]:
        i = x.index(x_c)
        scatter_foot_trace(f1[i], i/float(3), label="Kp = " + str(x_c))

    plt.legend(loc="upper right")
    plt.title("Locus of the front left foot with different stiffness values")
    plt.xlim([0.35, 1.6])
    plt.savefig(IMG)
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] == "process":
            data, data_config_fields = analysis.get_foot_data(FOLDER)
            with open(os.path.join(FOLDER, "locus.pkl"), "wb") as f:
                pickle.dump([data, data_config_fields], f, protocol=2)
            exit()

    [data, changing_config] = pickle.load(open(os.path.join(FOLDER, "locus.pkl"), "rb"))
    plot_locus(data)