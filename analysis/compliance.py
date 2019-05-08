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
plt.rc('xtick', color="#606060")
plt.rc('ytick', color="#606060")
plt.rc('figure', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=False)


FOLDER = "/home/gurbain/docker_sim/experiments/compliance"
IMG = "/home/gabs48/compliance.png"


def select_data(data):

    new_data = []
    for d in data:
            d['train_speed'] = max(0, 1 - abs(0.25 - d['train_speed']) / 0.25)
            d['cl_speed'] = max(0, 1 - abs(0.25 - d['cl_speed']) / 0.25)
            d['test_speed'] = max(0, 1 - abs(0.25 - d['test_speed']) / 0.25)
            new_data.append(d)

    return new_data


def aggregate_falls(data):

    new_data = []
    for d in data:
        if bool(d["cl_fall"]):
            d["test_fall"] = True
        if bool(d["train_fall"]):
            d["cl_fall"] = True
            d["test_fall"] = True

        new_data.append(d)

    return new_data


def plot(ax, data, field_x, field_y, phases=["train_", "test_"], j=0, ylim=None, title=None):

    leg = ["RCF Controller", "Neural Controller"]
    if "entropy" in field_y:
        leg = [field_y.replace("entropy_pred", "").replace("entropy_target", "")]
        #leg = ["Train " + field_y, "Test " + field_y]
    div = 1.0
    if field_y == "COT":
        div = 6.0
    if field_y == "entropy_target" or field_y == "entropy_pred":
        p = "train_" if field_y == "entropy_target" else "test_"
        for k, name in enumerate(["app", "sample", "spectral"]):
            plot(ax, data, field_x, field_y + "_" + name, phases=[p], j=k)
    else:
        for i, phase in enumerate(phases):
            x, y_av, y_std = analysis.get_graph_data(data, field_x, phase + field_y, "No Field")
            ax.plot(x, y_av, linestyle=analysis.get_lines(i + j),
                    linewidth=2, color=analysis.get_cols(i + j),
                    label=leg[i].replace('_', ' ').title())
            ax.fill_between(x, y_av - y_std / div, y_av + y_std / div,
                            alpha=0.2, edgecolor=analysis.get_cols(i + j),
                            facecolor=analysis.get_cols(i + j))

    ax.set_xlabel('Stiffness [N.m/rad]')
    ax.set_xlim([0, 4000])
    if ylim:
        ax.set_ylim(ylim)
    if title is None:
        title = str(field_y.replace('_', ' '))
    ax.set_title(title)
    ax.legend(loc="best")


def plot_compliance(data):

    field_x = 'physics_kp'
    fields_y = ["speed", "grf_step_len", "grf_steps",
                "nrmse", "entropy_target", "entropy_pred",
                "power", "grf_max"]
    titles = ["Robot Speed", "Step Length", "Number of Steps",
              "NRMSE", "RCF Entropy", "Neural Network Entropy",
              "Robot Power", "Maximum Ground Reaction Force"]

    fig_data = aggregate_falls(select_data(data))
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12), dpi=80)
    fig.subplots_adjust(left=0.04, bottom=0.04, right=0.96, top=0.96, wspace=0.25, hspace=0.4)
    k = 0
    for i in range(3):
        for j in range(3):
            ylim = None
            if k == 2:
                ylim = [20, 60]
            if k == 8:
                break
            plot(axes[i, j], fig_data, field_x, fields_y[k], ylim=ylim, title=titles[k])
            k += 1
    plt.savefig(IMG)
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] == "process":
            data, data_config_fields = analysis.get_data(FOLDER)
            with open(os.path.join(FOLDER, "compliance.pkl"), "wb") as f:
                pickle.dump([data, data_config_fields], f, protocol=2)
            exit()

    [data, changing_config] = pickle.load(open(os.path.join(FOLDER, "compliance.pkl"), "rb"))
    plot_compliance(data)