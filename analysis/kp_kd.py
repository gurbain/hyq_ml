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


def plot_kpkd(data):

    # Get data
    field_x = 'physics_init_impedance'
    field_y = 'test_x_speed'
    field_z = 'physics_noise'
    data = plot_metrics.get_graph_data(data, field_x, field_y, field_z)

    # Plot figure
    plt.figure(figsize=(10, 8), dpi=80)
    for j in range(len(data[3])):
        plt.plot(data[0], data[1][:, j], linestyle=plot_metrics.get_lines(j),
                 linewidth=2, color=plot_metrics.get_cols(j),
                 label=str(field_z).replace("_", " ") + " = " +
                       str(data[3][j]))
        plt.fill_between(data[0],
                         data[1][:, j] - data[2][:, j] / 5.0,
                         data[1][:, j] + data[2][:, j] / 5.0,
                         alpha=0.1, edgecolor=plot_metrics.get_cols(j),
                         facecolor=plot_metrics.get_cols(j))
        plt.title((str(field_y) + " depending on " + str(field_x) +
                   " with different " + str(field_z)).replace("_", " "))
    plt.legend()
    plt.show()


if __name__ == "__main__":

    data, data_config_fields = plot_metrics.get_data("/home/gurbain/docker_sim/experiments/kpkd")
    plot_kpkd(data)