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


FOLDER = "/home/gurbain/docker_sim/experiments/hl"


def select_data(data, sel="grf"):

    new_data = []
    for d in data:
        if sel == "grf" and eval(d["simulation_inputs"]) == ["bias", "grf"]:
            new_data.append(d)
        if sel == 'grf + joints' and eval(d["simulation_inputs"]) == ["bias", "grf", "joints"]:
            new_data.append(d)
        if sel == 'grf + imu' and eval(d["simulation_inputs"]) == ["bias", "grf", "imu"]:
            new_data.append(d)
        if sel == "all" and eval(d["simulation_inputs"]) == ["bias", "grf", "joints", "imu"]:
            new_data.append(d)
    return new_data


def plot(ax, data, field_x, field_y, field_z, label=None):

    data = plot_metrics.get_graph_data(data, field_x, field_y, field_z)
    ax.plot(data[0], data[1], linewidth=2, label=label)
    ax.fill_between(data[0], data[1] - data[2] / 5.0, data[1] + data[2] / 5.0, alpha=0.1)
    ax.set_title(str(field_y))


def plot_hl(data):

    field_x = 'physics_noise'
    field_z = 'No Field'
    fields_y = ["test_nrmse", "test_speed", "test_x_speed", "test_y_speed",
                "test_COT", "test_power", "train_average_computation_time", "test_grf_steps",
                "test_z_range", "test_pitch_range", "test_roll_range", "test_grf_max",
                "test_grf_step_len", "diff_dist",  "diff_x_speed", "diff_y_speed",
                "diff_power", "diff_COT", "diff_grf_step_len", "diff_grf_steps",
                "diff_grf_max", "diff_z_range", "diff_pitch_range", "diff_roll_range"
              ]

    # Plot figure
    plt.figure(figsize=(80, 60), dpi=80)
    for i, f in enumerate(fields_y):
        ax = plt.subplot(6, 4, i+1)
        #plot(ax, select_data(data, 'grf'), field_x, f, field_z, label='GRF')
        plot(ax, select_data(data, 'grf + joints'), field_x, f, field_z, label='GRF + Joints')
        #plot(ax, select_data(data, 'grf + imu'), field_x, f, field_z, label='GRF + IMU')
        plot(ax, select_data(data, 'all'), field_x, f, field_z, label='GRF + Joint + IMU')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] == "process":
            data, data_config_fields = plot_metrics.get_data(FOLDER)
            print data_config_fields
            with open(os.path.join(FOLDER, "hl.pkl"), "wb") as f:
                pickle.dump([data, data_config_fields], f, protocol=2)
            exit()

    [data, changing_config] = pickle.load(open(os.path.join(FOLDER, "hl.pkl"), "rb"))
    plot_hl(data)