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


EXP_FOLDER = "/home/gabs48/src/quadruped/hyq/hyq_ml/data/"


def get_cols(i):

    if 'axes.prop_cycle' in plt.rcParams:
        cols = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
        cols[2], cols[3] = cols[3], cols[2]

        return cols[i % len(cols)]


def get_blue():

    return get_cols(0)


def get_red():
    return get_cols(1)


def get_green():
    return get_cols(2)


def get_yellow():
    return get_cols(3)


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
    i = 0
    for a, b in tqdm(itertools.combinations(config_data, 2)):
        if i > 300000:
            break
        for key, value in a.iteritems():
            if key in b:
                if a[key] != b[key]:
                    if key not in changing_config:
                        changing_config.append(key)
            else:
                print " === ERROR: All the configuration files of the experiment " \
                      " directories must have the same fields!"
                return -1
        i += 1

    # Mix all in a big dictionary
    data = metrics_data
    for i, d in enumerate(data):
        for key in changing_config:
            c = config_data[i][key]
            if c.isdigit():
                d[key] = float(c)
            else:
                d[key] = str(c)

    # Add missing fields and cleanup others
    clean_data(data)

    return data, changing_config


def save_conf(folder, field_x, field_y, field_z):

    with open(os.path.join(os.path.dirname(folder), "plot_conf.pkl"), "wb") as f:
        pickle.dump([folder, field_x, field_y, field_z], f, protocol=2)


def get_fields(data, config_fields, conf):

    fields = sorted(data[0].keys())

    if conf is not None:
        f_x_mask = [f == conf[1] for f in config_fields]
        f_y_mask = [f == conf[2] for f in fields]
    else:
        f_x_mask = [False for _ in config_fields]
        f_y_mask = [False for _ in fields]

    f_x = config_fields[Picker(title="Select the Graph X-Axis (only one choice)",
                               options=config_fields, init_options=f_x_mask).getIndex()[0]]
    f_y_i = Picker(title="Select the Graph Y-Axis (only one choice)",
                        options=fields, init_options=f_y_mask).getIndex()

    f_y = []
    for i in f_y_i:
        f_y.append(fields[i])

    z_fields = ["yes", "no - average all"]
    z_set = []

    config_fields.remove(f_x)
    for z in config_fields:
        z_set.append(sorted(list(set([d[z] for d in data]))))

    for x in itertools.product(*z_set):
        opt = "no - select "
        for i, z in enumerate(x):
            opt += config_fields[i] + "=" + str(z) + " "
        z_fields.append(opt)

    if conf is not None:
        f_z_mask = [f == conf[3] for f in z_fields]
    else:
        f_z_mask = [False for _ in z_fields]

    f_z_i = Picker(title="Do you want to plot multiple graphs?",
                            options=z_fields, init_options=f_z_mask).getIndex()[0]

    if f_z_i == 0:
        f_z = config_fields[Picker(title="Select the Graph Z field",
                                   options=config_fields,
                                   init_options=[False for _ in config_fields]).getIndex()[0]]
    else:
        f_z = z_fields[f_z_i]

    return f_x, f_y, f_z


def clean_data(data):

    for d in data:
        d["test_grf_steps"] = (d["test_lh_grf_steps"] + d["test_lf_grf_steps"] +
                               d["test_rf_grf_steps"] + d["test_rh_grf_steps"]) / 4

        d["test_grf_step_len"] = (d["test_lh_grf_step_len"] + d["test_lf_grf_step_len"] +
                                  d["test_rf_grf_step_len"] + d["test_rh_grf_step_len"]) / 4

        d["cl_grf_steps"] = (d["cl_lh_grf_steps"] + d["cl_lf_grf_steps"] +
                               d["cl_rf_grf_steps"] + d["cl_rh_grf_steps"]) / 4

        d["cl_grf_step_len"] = (d["cl_lh_grf_step_len"] + d["cl_lf_grf_step_len"] +
                                  d["cl_rf_grf_step_len"] + d["cl_rh_grf_step_len"]) / 4

        d["train_grf_steps"] = (d["train_lh_grf_steps"] + d["train_lf_grf_steps"] +
                               d["train_rf_grf_steps"] + d["train_rh_grf_steps"]) / 4

        d["train_grf_step_len"] = (d["train_lh_grf_step_len"] + d["train_lf_grf_step_len"] +
                                  d["train_rf_grf_step_len"] + d["train_rh_grf_step_len"]) / 4

        for k in d.keys():
            if "y_speed" in k:
                d[k] = abs(d[k])

            if "roll_range" in k or "pitch_range" in k:
                d[k] = float(d[k]) % (2 * np.pi)

            if "physics_init_impedance" == k:
                imp = eval(d["physics_init_impedance"])
                if imp is None:
                    d["physics_kp"] = np.nan
                    d["physics_kd"] = np.nan
                else:
                    d["physics_kp"] = (imp[2] + imp[4]) / 2
                    d["physics_kd"] = (imp[3] + imp[5]) / 2
            if "entropy" in k:
                for i, name in enumerate(["perm", "svd", "app", "sample", "spectral"]):
                    d[k + "_" + name] = d[k][i]

        d["diff_dist"] = abs(d["test_dist"] - d["train_dist"])
        d["diff_speed"] = abs(d["test_speed"] - d["train_speed"])
        d["diff_nrmse"] = abs(d["test_nrmse"] - d["train_nrmse"])
        d["diff_x_speed"] = abs(d["test_x_speed"] - d["train_x_speed"])
        d["diff_y_speed"] = abs(d["test_y_speed"] - d["train_y_speed"])
        d["diff_power"] = abs(d["test_power"] - d["train_power"])
        d["diff_COT"] = abs(d["test_power"] - d["train_power"])
        d["diff_grf_step_len"] = abs(d["test_grf_step_len"] - d["train_grf_step_len"])
        d["diff_grf_steps"] = abs(d["test_grf_steps"] - d["train_grf_steps"])
        d["diff_grf_max"] = abs(d["test_grf_max"] - d["train_grf_max"])
        d["diff_z_range"] = abs(d["test_z_range"] - d["train_z_range"])
        d["diff_pitch_range"] = abs(d["test_pitch_range"] - d["train_pitch_range"])
        d["diff_roll_range"] = abs(d["test_roll_range"] - d["train_roll_range"])


def get_graph_data(data, field_x, field_y, field_z):

    if field_z != "No Field":
        x_list = [d[field_x] for d in data]
        x_set = sorted(list(set(x_list)))
        z_list = [d[field_z] for d in data]
        z_set = sorted(list(set(z_list)))
        n_sample = max(max([x_list.count(e) for e in x_set]),
                       max([z_list.count(e) for e in z_set]))
        y_val = np.empty((len(x_set), len(z_set), n_sample))
        y_val[:, :, :] = np.nan
        sampling_index = np.zeros((len(x_set), len(z_set)), dtype=np.int8)

        for d in data:
            x_index = x_set.index(d[field_x])
            z_index = z_set.index(d[field_z])
            y_val[x_index, z_index, sampling_index[x_index, z_index]] = d[field_y]
            sampling_index[x_index, z_index] += 1

        y_av = np.nanmean(y_val, axis=2)
        y_std = np.nanstd(y_val, axis=2)

        return np.array(x_set), y_av, y_std, z_set

    else:
        x_list = [d[field_x] for d in data]
        x_set = sorted(list(set(x_list)))
        n_sample = max([x_list.count(e) for e in x_set])

        y_val = np.empty((len(x_set), n_sample))
        y_val[:, :] = np.nan
        sampling_index = np.zeros((len(x_set)), dtype=np.int8)

        for d in data:
            x_index = x_set.index(d[field_x])
            y_val[x_index, sampling_index[x_index]] = d[field_y]
            sampling_index[x_index] += 1

        y_av = np.nanmean(y_val, axis=1)
        y_std = np.nanstd(y_val, axis=1)

        return np.array(x_set), y_av, y_std


def plot_graph(graph_data, field_x, field_y, field_z):

    x_scale = "linear"
    if 100 * (graph_data[0][1] - graph_data[0][0]) <= graph_data[0][-1] - graph_data[0][-2]:
        x_scale = "log"

    if z != "No Field":

        plt.figure(figsize=(10, 8), dpi=80)
        for j in range(len(graph_data[3])):
            plt.plot(graph_data[0], graph_data[1][:, j], linestyle=get_lines(j),
                     linewidth=2, color=get_cols(j),
                     label=str(field_z).replace("_", " ") + " = " +
                           str(graph_data[3][j]))
            plt.fill_between(graph_data[0],
                             graph_data[1][:, j] - graph_data[2][:, j]/5.0,
                             graph_data[1][:, j] + graph_data[2][:, j]/5.0,
                             alpha=0.1, edgecolor=get_cols(j), facecolor=get_cols(j))
        plt.title((str(field_y) + " depending on " + str(field_x) +
                  " with different " + str(field_z)).replace("_", " "))
        plt.xscale(x_scale)
        plt.legend()
        plt.show()

    else:

        plt.figure(figsize=(10, 8), dpi=80)
        plt.plot(graph_data[0], graph_data[1], linewidth=2)
        plt.fill_between(graph_data[0], graph_data[1] - graph_data[2]/5.0,
                         graph_data[1] + graph_data[2]/5.0, alpha=0.1)
        plt.title((str(field_y) + " depending on " +
                   str(field_x)).replace("_", " "))
        plt.xscale(x_scale)
        plt.legend()
        plt.show()


if __name__ == "__main__":

    folder, default_conf = get_folder()
    data, data_config_fields = get_data(folder)
    field_x, fields_y, field_z = get_fields(data, data_config_fields, default_conf)
    save_conf(folder, field_x, fields_y, field_z)
    graph_data = get_graph_data(data, field_x, fields_y, field_z)
    plot_graph(graph_data, field_x, fields_y, field_z)