import ConfigParser
import itertools
import matplotlib
import numpy as np
import os
import pickle
from tqdm import tqdm

from hyq.picker import *

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rc('text', usetex=True)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)


EXP_FOLDER = "/home/gabs48/src/quadruped/hyq/hyq_ml/data/docker_sim/"


def get_cols(i):

    if 'axes.prop_cycle' in plt.rcParams:
        cols = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
        cols[2], cols[3] = cols[3], cols[2]

        return cols[i % len(cols)]


def get_lines(i):

    lines = ["-", "--", ":", "-."]
    return lines[i % len(lines)]


def get_config_items(c):

    d = {}
    for sect in c.sections():
        for (key, val) in c.items(sect):
            d[str(sect).lower() + "_" + str(key).lower()] = val
    return d


def get_folder():

    # Find plotting config
    conf = None
    f = os.path.join(EXP_FOLDER, "plot_conf.pkl")
    if os.path.isfile(f):
         conf = pickle.load(open(f, "rb"))

    subdirs = [f for f in os.listdir(EXP_FOLDER) if os.path.isdir(os.path.join(EXP_FOLDER, f))]
    if conf is not None:
        subdirs_mask = [s == conf[0].split("/")[-1] for s in subdirs]
    else:
        subdirs_mask = [False for _ in subdirs]

    f = Picker(title="Select the experiment folder to analyze "
                     "(only the first will be plotted)",
               options=subdirs, init_options=subdirs_mask).getIndex()

    return EXP_FOLDER + subdirs[f[0]], conf


def get_data(folder):

    # Read the folder datas
    metrics_data = []
    config_data = []
    for subdir in tqdm(os.listdir(folder)):
        config_filename = os.path.join(os.path.join(folder, subdir), "config.txt")
        physics_filename = os.path.join(os.path.join(folder, subdir), "metrics.pkl")
        if os.path.isfile(config_filename) and os.path.isfile(physics_filename):
            config = ConfigParser.ConfigParser()
            config.read(config_filename)
            config_data.append(get_config_items(config))

            d = pickle.load(open(physics_filename, "rb"))
            metrics_data.append(d)

    # Find changing parameter in config
    changing_config = []
    for a, b in itertools.combinations(config_data, 2):
        for key, value in a.iteritems():
            if key in b:
                if a[key] != b[key]:
                    if key not in changing_config:
                        changing_config.append(key)
            else:
                print " === ERROR: All the configuration files of the experiment " \
                      " directories must have the same fields!"
                return -1

    # Mix all in a big dictionary
    data = metrics_data
    for i, d in enumerate(data):
        for key in changing_config:
            d[key] = float(config_data[i][key])

    return data


def save_conf(folder, field_x, field_y, field_z):

    with open(os.path.join(os.path.dirname(folder), "plot_conf.pkl"), "wb") as f:
        pickle.dump([folder, field_x, field_y, field_z], f, protocol=2)


def get_fields(data, conf):

    fields = sorted(data[0].keys())
    z_fields = ["No Field"] + fields

    if conf is not None:
        f_x_mask = [f == conf[1] for f in fields]
        f_y_mask = [f == conf[2] for f in fields]
        f_z_mask = [f == conf[3] for f in z_fields]
    else:
        f_x_mask = [False for _ in fields]
        f_y_mask = [False for _ in fields]
        f_z_mask = [False for _ in z_fields]

    f_x = fields[Picker(title="Select the Graph X-Axis (only one choice)",
                        options=fields, init_options=f_x_mask).getIndex()[0]]
    f_z = z_fields[Picker(title="Select the Field for Multiple Graphs (only one choice)",
                          options=z_fields, init_options=f_z_mask).getIndex()[0]]
    f_y = fields[Picker(title="Select the Graph Y-Axis (only one choice)",
                        options=fields, init_options=f_y_mask).getIndex()[0]]

    return f_x, f_y, f_z


def get_graph_data(data, field_x, field_y, field_z):

    if z != "No Field":
        x_list = [d[field_x] for d in data]
        x_set = sorted(list(set(x_list)))
        z_list = [d[field_z] for d in data]
        z_set = sorted(list(set(z_list)))
        n_sample = max(max([x_list.count(e) for e in x_set]),
                       max([z_list.count(e) for e in z_set]))

        y_val = np.empty((len(x_set), len(z_set), n_sample))
        sampling_index = np.zeros((len(x_set), len(z_set)), dtype=np.int8)

        for d in data:
            x_index = x_set.index(d[field_x])
            z_index = z_set.index(d[field_z])
            y_val[x_index, z_index, sampling_index[x_index, z_index]] = d[field_y]
            sampling_index[x_index, z_index] += 1

        y_av = np.nanmean(y_val, axis=2)
        y_std = np.nanstd(y_val, axis=2)

        return x_set, z_set, y_av, y_std

    else:
        x_list = [d[field_x] for d in data]
        x_set = sorted(list(set(x_list)))
        n_sample = max([x_list.count(e) for e in x_set])

        y_val = np.empty((len(x_set), n_sample))
        sampling_index = np.zeros((len(x_set)), dtype=np.int8)

        for d in data:
            x_index = x_set.index(d[field_x])
            y_val[x_index, sampling_index[x_index]] = d[field_y]
            sampling_index[x_index] += 1

        y_av = np.nanmean(y_val, axis=1)
        y_std = np.nanstd(y_val, axis=1)

        return np.array(x_set), y_av, y_std


def plot_graph(graph_data, field_x, field_y, field_z):

    if z != "No Field":

        plt.figure(figsize=(10, 8), dpi=80)
        for j in range(len(graph_data[1])):
            plt.semilogx(graph_data[0], graph_data[2][:, j], linestyle=get_lines(j),
                         linewidth=2, color=get_cols(j),
                         label=str(field_z).replace("_", " ") + " = " +
                               str(graph_data[1][j]))
            plt.fill_between(graph_data[0],
                             graph_data[2][:, j] - graph_data[3][:, j],
                             graph_data[2][:, j] + graph_data[3][:, j],
                             alpha=0.1, edgecolor=get_cols(j), facecolor=get_cols(j))
        plt.title((str(field_y) + " depending on " + str(field_x) +
                  " with different " + str(field_z)).replace("_", " "))
        plt.legend()
        plt.show()

    else:

        plt.figure(figsize=(10, 8), dpi=80)
        plt.semilogx(graph_data[0], graph_data[1], linewidth=2)
        plt.fill_between(graph_data[0], graph_data[1] - graph_data[2],
                         graph_data[1] + graph_data[2], alpha=0.1)
        plt.title((str(field_y) + " depending on " +
                   str(field_x)).replace("_", " "))
        plt.legend()
        plt.show()


if __name__ == "__main__":

    f, c = get_folder()
    d = get_data(f)
    x, y, z = get_fields(d, c)
    save_conf(f, x, y, z)
    g = get_graph_data(d, x, y, z)
    plot_graph(g, x, y, z)