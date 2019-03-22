import os
import numpy as np
import pickle
import sys

import matplotlib.colors as cols
import matplotlib.pyplot as plt

from hyq import analysis

plt.style.use('fivethirtyeight')
plt.rc('text', usetex=False)
plt.rc('font', size=9)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)


FOLDER = "/home/gurbain/docker_sim/experiments/compliance"


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


def plot(ax, data, field_x, field_y, phases=["train_", "test_"], j=0):

    leg = ["Training Phase", "Testing Phase"]
    if "entropy" in field_y:
        leg = [field_y]
    div = 1.0
    if field_y == "COT":
        div = 6.0
    if field_y == "entropy_target":
        for k, name in enumerate(["perm", "svd", "app", "sample", "spectral"]):
            plot(ax, data, field_x, field_y + "_" + name, ["train_"], k)
    else:
        for i, phase in enumerate(phases):
            x, y_av, y_std = analysis.get_graph_data(data, field_x, phase + field_y, "No Field")
            ax.plot(x, y_av, linestyle=analysis.get_lines(i + j),
                    linewidth=2, color=analysis.get_cols(i + j),
                    label=leg[i])
            ax.fill_between(x, y_av - y_std / div, y_av + y_std / div,
                            alpha=0.2, edgecolor=analysis.get_cols(i + j),
                            facecolor=analysis.get_cols(i + j))

    ax.set_xlabel('Stiffness [N.m/rad]')
    ax.set_title(str(field_y))
    ax.legend(loc="best")


def plot_compliance(data):

    field_x = 'physics_kp'
    fields_y = ["nrmse", "speed", "power", "grf_max",
                "entropy_target", "grf_step_len"]

    fig_data = aggregate_falls(select_data(data))
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12), dpi=80)
    k = 0
    for i in range(3):
        for j in range(2):
            plot(axes[i, j], fig_data, field_x, fields_y[k])
            k += 1
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