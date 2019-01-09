
import ConfigParser
import matplotlib
import numpy as np
import os
import pickle
from tqdm import tqdm


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rc('text', usetex=True)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)


EXP_FOLDER = "/home/gabs48/src/quadruped/hyq/hyq_ml/data/docker_sim/kpkd"


def get_cols(i):

    if 'axes.prop_cycle' in plt.rcParams:
        cols = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
        cols[2], cols[3] = cols[3], cols[2]

    return cols[i % len(cols)]


def get_lines(i):

    lines = ["-", "--", ":", "-."]
    return lines[i % len(lines)]


def get_data():

    data = []
    for subdir in tqdm(os.listdir(EXP_FOLDER)):

        config_filename = os.path.join(os.path.join(EXP_FOLDER, subdir), "config.txt")
        physics_filename = os.path.join(os.path.join(EXP_FOLDER, subdir), "metrics.pkl")
        if os.path.isfile(config_filename) and os.path.isfile(physics_filename):

                config = ConfigParser.ConfigParser()
                config.read(config_filename)
                init_impedance = eval(config.get("Physics", "init_impedance"))
                tsim = int(config.get("Timing", "t_sim"))
                kp = init_impedance[0]
                kd = init_impedance[1]

                metrics_data = pickle.load(open(physics_filename, "rb"))
                metrics_data["Kp"] = kp
                metrics_data["Kd"] = kd
                metrics_data["t_sim"] = tsim

                data.append(metrics_data)

    return data


def plot_data(data):

    # Get the Kp-Kd space
    kd_set = sorted(list(set(d["Kd"] for d in data)))
    kp_set = sorted(list(set(d["Kp"] for d in data)))
    n_sample = len(data) / len(kd_set) / len(kp_set)

    # Get final x,y distance for each robot
    x_dist = np.zeros((len(kp_set), len(kd_set), n_sample))
    y_dist = np.zeros((len(kp_set), len(kd_set), n_sample))
    x_vel = np.zeros((len(kp_set), len(kd_set), n_sample))
    y_vel = np.zeros((len(kp_set), len(kd_set), n_sample))
    sampling_index = np.zeros((len(kp_set), len(kd_set)), dtype=np.int8)
    for d in data:
        kd_index = kd_set.index(d["Kd"])
        kp_index = kp_set.index(d["Kp"])
        x_dist[kp_index, kd_index, sampling_index[kp_index, kd_index]] = d["x_range"]
        y_dist[kp_index, kd_index, sampling_index[kp_index, kd_index]] = d["y_range"]
        x_vel[kp_index, kd_index, sampling_index[kp_index, kd_index]] = d["x_range"] / d["t_sim"]
        y_vel[kp_index, kd_index, sampling_index[kp_index, kd_index]] = d["y_range"] / d["t_sim"]
        sampling_index[kp_index, kd_index] += 1

    x_dist_av = np.mean(x_dist, axis=2)
    y_dist_av = np.mean(y_dist, axis=2)
    x_dist_std = np.std(x_dist, axis=2)
    y_dist_std = np.std(y_dist, axis=2)
    x_vel_av = np.mean(x_vel, axis=2)
    y_vel_av = np.mean(y_vel, axis=2)
    x_vel_std = np.std(x_vel, axis=2)
    y_vel_std = np.std(y_vel, axis=2)

    # Plot X distance
    plt.figure(figsize=(10, 8), dpi=80)
    print len(kp_set), len(x_dist_av[:, 0])
    for j in range(len(kd_set)):
        plt.semilogx(kp_set, x_dist_av[:, j], linestyle=get_lines(j),
                 linewidth=2, color=get_cols(j), label="Kd = " + str(kd_set[j]))
        plt.fill_between(kp_set,
                         x_dist_av[:, j] - x_dist_std[:, j],
                         x_dist_av[:, j] + x_dist_std[:, j],
                         alpha=0.2, edgecolor=get_cols(j), facecolor=get_cols(j))
    plt.title("X distance depending on Kp")
    plt.legend()
    plt.show()

    # Plot Y distance
    plt.figure(figsize=(10, 8), dpi=80)
    for j in range(len(kd_set)):
        plt.plot(kp_set, y_dist_av[:, j], linestyle=get_lines(j),
                 linewidth=2, color=get_cols(j), label="Kd = " + str(kd_set[j]))
        plt.fill_between(kp_set,
                         y_dist_av[:, j] - y_dist_std[:, j],
                         y_dist_av[:, j] + y_dist_std[:, j],
                         alpha=0.2, edgecolor=get_cols(j), facecolor=get_cols(j))
    plt.title("Y distance depending on Kp")
    plt.legend()
    plt.show()

    # Plot X speed
    plt.figure(figsize=(10, 8), dpi=80)
    for j in range(len(kd_set)):
        plt.plot(kp_set, x_vel_av[:, j], linestyle=get_lines(j),
                 linewidth=2, color=get_cols(j), label="Kd = " + str(kd_set[j]))
        plt.fill_between(kp_set,
                         x_vel_av[:, j] - x_vel_std[:, j],
                         x_vel_av[:, j] + x_vel_std[:, j],
                         alpha=0.2, edgecolor=get_cols(j), facecolor=get_cols(j))
    plt.title("X speed depending on Kp")
    plt.legend()
    plt.show()

    # Plot Y speed
    plt.figure(figsize=(10, 8), dpi=80)
    for j in range(len(kd_set)):
        plt.plot(kp_set, y_vel_av[:, j], linestyle=get_lines(j),
                 linewidth=2, color=get_cols(j), label="Kd = " + str(kd_set[j]))
        plt.fill_between(kp_set,
                         y_vel_av[:, j] - y_vel_std[:, j],
                         y_vel_av[:, j] + y_vel_std[:, j],
                         alpha=0.2, edgecolor=get_cols(j), facecolor=get_cols(j))
    plt.title("Y speed depending on Kp")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    d = get_data()
    plot_data(d)
