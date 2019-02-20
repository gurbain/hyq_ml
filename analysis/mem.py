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


def plot_mem(data):

    # Get data
    field_x = 'force_delay_line_n'
    field_y = 'test_x_speed'
    field_z = 'physics_noise'
    data = plot_metrics.get_graph_data(data, field_x, field_y, field_z)

    # Plot figure
    plt.figure(figsize=(10, 8), dpi=80)
    for j in range(len(data[3])):
        plt.plot(data[0], data[1][:, j], linestyle=get_lines(j),
                 linewidth=2, color=get_cols(j),
                 label=str(field_z).replace("_", " ") + " = " +
                       str(data[3][j]))
        plt.fill_between(data[0],
                         data[1][:, j] - data[2][:, j] / 5.0,
                         data[1][:, j] + data[2][:, j] / 5.0,
                         alpha=0.1, edgecolor=get_cols(j), facecolor=get_cols(j))
        plt.title((str(field_y) + " depending on " + str(field_x) +
                   " with different " + str(field_z)).replace("_", " "))
    plt.legend()
    plt.show()


if __name__ == "__main__":

    data, data_config_fields = plot_metrics.get_data("/home/gurbain/docker_sim/experiments/mem")
    plot_mem_nl(data)