import copy
import os
import numpy as np
import pickle
import sys
from scipy.interpolate import griddata, interp2d, interp1d

import matplotlib.colors as cols
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid

import plot_metrics

plt.style.use('fivethirtyeight')
plt.rc('text', usetex=False)
plt.rc('font', size=9)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)


FOLDER = "/home/gurbain/docker_sim/experiments/kpkd"


def select_data(data):

    new_data = []
    for d in data:
        if not bool(d["test_fall"]) or bool(d["cl_fall"]):
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


def get_fall_data(data, field_x, field_y, field_z):

    data = plot_metrics.get_graph_data(data, field_x, field_y, field_z)

    fall_x = []
    fall_y = []
    not_fall_x = []
    not_fall_y = []
    for i, x in enumerate(data[0]):
        for j, y in enumerate(data[3]):
            if data[1][i, j] > 0.5:
                fall_x.append(x)
                fall_y.append(y)
            else:
                not_fall_x.append(x)
                not_fall_y.append(y)

    return fall_x, fall_y, not_fall_x, not_fall_y


def get_3d_data(data, field_x, field_y, field_z):

    x_set, z_av_tab, z_std_tab, y_set = plot_metrics.get_graph_data(data, field_x, field_y, field_z)
    x = []
    y = []
    z = []
    for i, xs in enumerate(x_set):
        for j, ys in enumerate(y_set):
            x.append(xs)
            y.append(ys)
            z.append(z_av_tab[i, j])

    return x, y, z


def plot(ax, data, field_x, field_y, field_z):

    data = plot_metrics.get_graph_data(data, field_x, field_y, field_z)
    for j in range(len(data[3])):
        ax.plot(data[0], data[1][:, j], linestyle=plot_metrics.get_lines(j),
                linewidth=2, color=plot_metrics.get_cols(j),
                label=str(field_z).replace("_", " ") + " = " +
                       str(data[3][j]))
        ax.fill_between(data[0],
                        data[1][:, j] - data[2][:, j] / 5.0,
                        data[1][:, j] + data[2][:, j] / 5.0,
                        alpha=0.1, edgecolor=plot_metrics.get_cols(j),
                        facecolor=plot_metrics.get_cols(j))
        ax.set_title(str(field_y))


def scatter_x_y(ax, data, field_x, field_y, field_z):

    x_set, z_av_tab, z_std_tab, y_set = plot_metrics.get_graph_data(data, field_x, field_y, field_z)
    if "grf_max" in field_y:
        im = ax.pcolormesh(x_set, y_set, z_av_tab.T,
                           norm=cols.LogNorm(vmin=z_av_tab.min(), vmax=z_av_tab.max()))
    else:
        im = ax.pcolormesh(x_set, y_set, z_av_tab.T)

    # x, y, z = get_3d_data(data, field_x, field_y, field_z)
    # logx = np.log10(x)
    # logy = np.log10(y)
    # logz = np.log10(z)
    # f = interp2d(logx, logy, logz, kind="linear")
    # xi = np.linspace(min(logx), max(logx), 100)
    # yi = np.linspace(min(logy), max(logy), 100)
    # zi = np.power(10, f(xi, yi))
    # xi = np.power(10, xi)
    # yi = np.power(10, yi)
    # im = ax.pcolormesh(xi, yi, zi)
    ks = np.logspace(np.log10(min(x_set)), np.log10(max(x_set)), 100)
    for j, kappa in enumerate([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1,
                               1, 10, 100, 1000, 10000]):
        ax.plot(ks, kappa * np.sqrt(ks), linestyle="--",
                linewidth=1, color="k", label="Kappa = " + str(kappa))
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([min(x_set), max(x_set)])
    ax.set_ylim([min(y_set), max(y_set)])
    ax.set_xlabel('Stiffness [N.m/rad]')
    ax.set_ylabel('Damping [N.m.s/rad]')
    ax.set_title(field_y.replace("_", " ").capitalize())

    return im


def plot_fall(ax, data, field_x, field_y, field_z):

    xf, yf, xnf, ynf = get_fall_data(data, field_x, field_y, field_z)
    ax.scatter(xf, yf, marker='x', facecolors=plot_metrics.get_red(), s=10)
    ax.scatter(xnf, ynf, marker='o', facecolors=plot_metrics.get_green(), s=20)
    ks = np.logspace(np.log10(min(min(xf), min(xnf))),
                     np.log10(max(max(xf), max(xnf))), 100)
    for j, kappa in enumerate([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1,
                               1, 10, 100, 1000, 10000]):
        ax.plot(ks, kappa * np.sqrt(ks), linestyle="--",
                linewidth=1, color="k", label="Kappa = " + str(kappa))

    ax.set_xlabel('Stiffness [N.m/rad]')
    ax.set_ylabel('Damping [N.m.s/rad]')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([min(min(xf), min(xnf)), max(max(xf), max(xnf))])
    ax.set_ylim([min(min(yf), min(ynf)), max(max(yf), max(ynf))])
    if field_y == "train_fall":
        ax.set_title("Training Falling Range")
    elif field_y == "cl_fall":
        ax.set_title("Closing Falling Range")
    else:
        ax.set_title("Test Falling Range")


def plot_kpkd(data):

    field_x = 'physics_kp'
    field_z = 'physics_kd'
    fields_y_1 = ["fall", "speed", "z_range", "pitch_range", "roll_range"]
    fields_y_2 = ["power", "COT", "grf_max", "grf_step_len", "grf_steps"]

    fig_data = aggregate_falls(data)
    fig, axes = plt.subplots(nrows=len(fields_y_1), ncols=4, figsize=(12, 36), dpi=80,
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})
    for j, f in enumerate(fields_y_1):
        for i, phase in enumerate(["train_", "cl_", "test_"]):
            if f == "fall":
                plot_fall(axes[j, i], fig_data, field_x, phase + f, field_z)
            else:
                im = scatter_x_y(axes[j, i], fig_data, field_x, phase + f, field_z)
        if 'im' in locals():
            if im is not None:
                fig.colorbar(im, cax=axes[j, 3])
    plt.show()

    fig, axes = plt.subplots(nrows=len(fields_y_2), ncols=4, figsize=(12, 36), dpi=60,
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})
    for j, f in enumerate(fields_y_2):
        for i, phase in enumerate(["train_", "cl_", "test_"]):
            if f == "fall":
                plot_fall(axes[j, i], fig_data, field_x, phase + f, field_z)
            else:
                im = scatter_x_y(axes[j, i], fig_data, field_x, phase + f, field_z)
        if 'im' in locals():
            if im is not None:
                fig.colorbar(im, cax=axes[j, 3])
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] == "process":
            data, data_config_fields = plot_metrics.get_data(FOLDER)
            with open(os.path.join(FOLDER, "kpkd.pkl"), "wb") as f:
                pickle.dump([data, data_config_fields], f, protocol=2)
            exit()

    [data, changing_config] = pickle.load(open(os.path.join(FOLDER, "kpkd.pkl"), "rb"))
    plot_kpkd(data)