import matplotlib
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rc('text', usetex=True)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)


FILENAME = "/home/gabs48/src/quadruped/hyq/hyq_ml/data/stifness_space_exploration/20181030-184700/results.pkl"

def get_cols():

    if 'axes.prop_cycle' in plt.rcParams:
        cols = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
        cols[2], cols[3] = cols[3], cols[2]
    return cols


def get_lines():

    return ["-", "--", ":", "-."]


def plot_data():
    # Retrieve data from file
    if os.path.exists(FILENAME):
        data = pickle.load(open(FILENAME, "rb"))

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
        x_dist[kp_index, kd_index, sampling_index[kp_index, kd_index]] = d["x"][-1]
        y_dist[kp_index, kd_index, sampling_index[kp_index, kd_index]] = d["y"][-1]
        x_vel[kp_index, kd_index, sampling_index[kp_index, kd_index]] = \
            d["x"][-1] / (d["t_sim"] - d["t_trot"])
        y_vel[kp_index, kd_index, sampling_index[kp_index, kd_index]] = \
            d["y"][-1] / (d["t_sim"] - d["t_trot"])
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
    for j in range(len(kd_set)):
        plt.plot(kp_set, x_dist_av[:, j], linestyle=get_lines()[j],
                 linewidth=2, color=get_cols()[j], label="Kd = " + str(kd_set[j]))
        plt.fill_between(kp_set,
                         x_dist_av[:, j] - x_dist_std[:, j],
                         x_dist_av[:, j] + x_dist_std[:, j],
                         alpha=0.2, edgecolor=get_cols()[j], facecolor=get_cols()[j])
    plt.title("X distance depending on Kp")
    plt.legend()
    plt.show()

    # Plot Y distance
    plt.figure(figsize=(10, 8), dpi=80)
    for j in range(len(kd_set)):
        plt.plot(kp_set, y_dist_av[:, j], linestyle=get_lines()[j],
                 linewidth=2, color=get_cols()[j], label="Kd = " + str(kd_set[j]))
        plt.fill_between(kp_set,
                         y_dist_av[:, j] - y_dist_std[:, j],
                         y_dist_av[:, j] + y_dist_std[:, j],
                         alpha=0.2, edgecolor=get_cols()[j], facecolor=get_cols()[j])
    plt.title("X distance depending on Kp")
    plt.legend()
    plt.show()

    # Plot X speed
    plt.figure(figsize=(10, 8), dpi=80)
    for j in range(len(kd_set)):
        plt.plot(kp_set, x_vel_av[:, j], linestyle=get_lines()[j],
                 linewidth=2, color=get_cols()[j], label="Kd = " + str(kd_set[j]))
        plt.fill_between(kp_set,
                         x_vel_av[:, j] - x_vel_std[:, j],
                         x_vel_av[:, j] + x_vel_std[:, j],
                         alpha=0.2, edgecolor=get_cols()[j], facecolor=get_cols()[j])
    plt.title("X velocity depending on Kp")
    plt.legend()
    plt.show()

    # Plot Y speed
    plt.figure(figsize=(10, 8), dpi=80)
    for j in range(len(kd_set)):
        plt.plot(kp_set, y_vel_av[:, j], linestyle=get_lines()[j],
                 linewidth=2, color=get_cols()[j], label="Kd = " + str(kd_set[j]))
        plt.fill_between(kp_set,
                         y_vel_av[:, j] - y_vel_std[:, j],
                         y_vel_av[:, j] + y_vel_std[:, j],
                         alpha=0.2, edgecolor=get_cols()[j], facecolor=get_cols()[j])
    plt.title("X speed depending on Kp")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    plot_data()