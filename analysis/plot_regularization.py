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


EXP_FOLDER = "/home/gabs48/src/quadruped/hyq/hyq_ml/data/docker_sim/regularization"


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

            metrics_data = pickle.load(open(physics_filename, "rb"))
            metrics_data["reg"] = float(config.get("Network", "regularization"))

            data.append(metrics_data)

    return data


def clean_data(data):

    # Get the space
    reg_set = sorted(list(set(d["reg"] for d in data)))
    n_sample = len(data) / len(reg_set)

    # Get final x,y distance for each robot
    train_x_vel = np.zeros((len(reg_set), n_sample))
    cl_x_vel = np.zeros((len(reg_set), n_sample))
    test_x_vel = np.zeros((len(reg_set), n_sample))
    sampling_index = [0] * len(reg_set)

    for d in data:
        reg_index = reg_set.index(d["reg"])
        test_x_vel[reg_index, sampling_index[reg_index]] = d["test_x_range"] / d["t_test"]
        cl_x_vel[reg_index, sampling_index[reg_index]] = d["cl_x_range"] / d["t_cl"]
        train_x_vel[reg_index, sampling_index[reg_index]] = d["train_x_range"] / d["t_train"]
        sampling_index[reg_index] += 1

    train_x_vel_av = np.mean(train_x_vel, axis=1)
    train_x_vel_std = np.std(train_x_vel, axis=1)
    cl_x_vel_av = np.mean(cl_x_vel, axis=1)
    cl_x_vel_std = np.std(cl_x_vel, axis=1)
    test_x_vel_av = np.mean(test_x_vel, axis=1)
    test_x_vel_std = np.std(test_x_vel, axis=1)

    return reg_set, test_x_vel_av, test_x_vel_std, \
           train_x_vel_av, train_x_vel_std, \
           cl_x_vel_av, cl_x_vel_std


def plot_data(datas):

    reg_set, test_x_vel_av, test_x_vel_std, \
    train_x_vel_av, train_x_vel_std, \
    cl_x_vel_av, cl_x_vel_std = datas

    # -- X Velocity

    # Train
    plt.semilogx(reg_set, train_x_vel_av, linestyle=get_lines(0), linewidth=2,
                 color=get_cols(0), label="Train")
    plt.fill_between(reg_set,
                     train_x_vel_av - train_x_vel_std,
                     train_x_vel_av + train_x_vel_std,
                     alpha=0.2, edgecolor=get_cols(0), facecolor=get_cols(0))

    # Closing the loop
    plt.semilogx(reg_set, cl_x_vel_av, linestyle=get_lines(1), linewidth=2,
                 color=get_cols(1), label="CL")
    plt.fill_between(reg_set,
                     cl_x_vel_av - cl_x_vel_std,
                     cl_x_vel_av + cl_x_vel_std,
                     alpha=0.2, edgecolor=get_cols(1), facecolor=get_cols(1))

    # Test
    plt.semilogx(reg_set, test_x_vel_av, linestyle=get_lines(2), linewidth=2,
                 color=get_cols(2), label="Test")
    plt.fill_between(reg_set,
                     test_x_vel_av - test_x_vel_std,
                     test_x_vel_av + test_x_vel_std,
                     alpha=0.2, edgecolor=get_cols(2), facecolor=get_cols(2))

    plt.title("X velocities depending on NN Regularization")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    d = get_data()
    e = clean_data(d)
    plot_data(e)