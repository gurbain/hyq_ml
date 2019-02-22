import os
import numpy as np
import pickle
import sys
from scipy.interpolate import griddata

import matplotlib.colors as cols
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import plot_metrics

plt.style.use('fivethirtyeight')
plt.rc('text', usetex=False)
plt.rc('font', size=9)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)


FOLDER = "/home/gurbain/docker_sim/experiments/mem"


def select_data(data, type_sel="with_joint"):

    new_data = []
    i = 0
    if type_sel == "with_joint":
        for d in data:
            if 'joints' in eval(d['simulation_inputs']):
                if not bool(d["test_fall"]) or bool(d["cl_fall"]):
                    i += 1
                    new_data.append(d)

    if type_sel == "no_joint":
        for d in data:
            if 'joints' not in eval(d['simulation_inputs']):
                if not bool(d["test_fall"]) or bool(d["cl_fall"]):
                    i += 1
                    new_data.append(d)

    return new_data


def get_data_points(data):

    return data


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


def plot_mem(data):

    field_x = 'force_delay_line_n'
    field_z = 'physics_noise'
    fields_y = ["test_nrmse", "test_speed", "test_x_speed", "test_y_speed",
                "test_COT", "test_power", "train_average_computation_time", "test_grf_steps",
                "test_z_range", "test_pitch_range", "test_roll_range", "test_grf_max",
                "test_grf_step_len", "diff_dist",  "diff_x_speed", "diff_y_speed",
                "diff_power", "diff_COT", "diff_grf_step_len", "diff_grf_steps",
                "diff_grf_max", "diff_z_range", "diff_pitch_range", "diff_roll_range"
              ]

    # Plot figure
    fig_data = select_data(data)
    plt.figure(figsize=(80, 60), dpi=80)
    for i, f in enumerate(fields_y):
        ax = plt.subplot(6, 4, i+1)
        plot(ax, fig_data, field_x, f, field_z)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] == "process":
            data, data_config_fields = plot_metrics.get_data(FOLDER)
            with open(os.path.join(os.path.dirname(FOLDER), "mem.pkl"), "wb") as f:
                pickle.dump([data, data_config_fields], f, protocol=2)
            exit()

    [data, changing_config] = pickle.load(open(os.path.join(os.path.dirname(FOLDER), "mem.pkl"), "rb"))
    plot_mem(data)