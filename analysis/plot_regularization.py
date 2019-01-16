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
#EXP_FOLDER = "/home/gabs48/test"

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
            metrics_data["reg"] = float(config.get("Force", "regularization"))
            metrics_data["buf"] = float(config.get("Force", "delay_line_n"))

            data.append(metrics_data)

    return data


def clean_data(data):

    # Get the space
    reg_set = sorted(list(set(d["reg"] for d in data)))
    buf_set = sorted(list(set(d["buf"] for d in data)))
    n_sample = len(data) / len(reg_set) / len(buf_set)
    # Get final x,y distance for each robot
    train_x_vel = np.zeros((len(reg_set), len(buf_set), n_sample))
    cl_x_vel = np.zeros((len(reg_set), len(buf_set), n_sample))
    test_x_vel = np.zeros((len(reg_set), len(buf_set), n_sample))
    sampling_index = np.zeros((len(reg_set), len(buf_set)), dtype=np.int8)

    for d in data:
        reg_index = reg_set.index(d["reg"])
        buf_index = buf_set.index(d["buf"])
        test_x_vel[reg_index, buf_index, sampling_index[reg_index, buf_index]] = d["test_x_range"] / d["t_test"]
        cl_x_vel[reg_index, buf_index, sampling_index[reg_index, buf_index]] = d["cl_x_range"] / d["t_cl"]
        train_x_vel[reg_index, buf_index, sampling_index[reg_index, buf_index]] = d["train_x_range"] / d["t_train"]
        sampling_index[reg_index, buf_index] += 1

    train_x_vel_av = np.mean(train_x_vel, axis=2)
    train_x_vel_std = np.std(train_x_vel, axis=2)
    cl_x_vel_av = np.mean(cl_x_vel, axis=2)
    cl_x_vel_std = np.std(cl_x_vel, axis=2)
    test_x_vel_av = np.mean(test_x_vel, axis=2)
    test_x_vel_std = np.std(test_x_vel, axis=2)

    return reg_set, buf_set, test_x_vel_av, test_x_vel_std, \
           train_x_vel_av, train_x_vel_std, \
           cl_x_vel_av, cl_x_vel_std


def plot_data(datas):

    reg_set, buf_set, test_x_vel_av, test_x_vel_std, \
    train_x_vel_av, train_x_vel_std, \
    cl_x_vel_av, cl_x_vel_std = datas

    # -- X Velocity

    # Train
    plt.figure(figsize=(10, 8), dpi=80)
    for j in range(len(buf_set)):
        plt.semilogx(reg_set, train_x_vel_av[:, j], linestyle=get_lines(j),
                 linewidth=2, color=get_cols(j), label="Memory Buffer = " + str(buf_set[j]))
        plt.fill_between(reg_set,
                         train_x_vel_av[:, j] - train_x_vel_std[:, j],
                         train_x_vel_av[:, j] + train_x_vel_std[:, j],
                         alpha=0.2, edgecolor=get_cols(j), facecolor=get_cols(j))
    plt.title("X Velocity during Training depending on Regularization with different memory buffers")
    plt.legend()
    plt.show()

    # Closing the loop
    plt.figure(figsize=(10, 8), dpi=80)
    for j in range(len(buf_set)):
        plt.semilogx(reg_set, cl_x_vel_av[:, j], linestyle=get_lines(j),
                 linewidth=2, color=get_cols(j), label="Memory Buffer = " + str(buf_set[j]))
        plt.fill_between(reg_set,
                         cl_x_vel_av[:, j] - cl_x_vel_std[:, j],
                         cl_x_vel_av[:, j] + cl_x_vel_std[:, j],
                         alpha=0.2, edgecolor=get_cols(j), facecolor=get_cols(j))
        plt.title("X Velocity during Loop-closing depending on Regularization with different memory buffers")
    plt.legend()
    plt.show()

    # Test
    plt.figure(figsize=(10, 8), dpi=80)
    for j in range(len(buf_set)):
        plt.semilogx(reg_set, test_x_vel_av[:, j], linestyle=get_lines(j),
                 linewidth=2, color=get_cols(j), label="Memory Buffer = " + str(buf_set[j]))
        plt.fill_between(reg_set,
                         test_x_vel_av[:, j] - test_x_vel_std[:, j],
                         test_x_vel_av[:, j] + test_x_vel_std[:, j],
                         alpha=0.2, edgecolor=get_cols(j), facecolor=get_cols(j))
    plt.title("X Velocity during Testing depending on Regularization with different memory buffers")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    d = get_data()
    e = clean_data(d)
    plot_data(e)