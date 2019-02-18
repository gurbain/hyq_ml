import os
import numpy as np
import pickle
import sys

import plot_metrics

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('fivethirtyeight')
plt.rc('text', usetex=False)
plt.rc('font', size=9)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)


def get_data(data, field_x, field_y):

    x_good = []
    y_good = []
    x_touch = []
    y_touch = []
    x_tilt = []
    y_tilt = []
    for d in data:
        # Cleanup
        if field_y == 'test_pitch_range':
            if d[field_y] > 5:
                d[field_y] = d[field_y] - 2 * np.pi
        if field_y == 'test_roll_range':
            if d[field_y] > 5:
                d[field_y] = d[field_y] - 2 * np.pi
        if field_y == 'test_nrmse':
            if d[field_y] > 0.27:
                d[field_y] = d[field_y] - 0.27
        if field_y == 'test_x_speed':
            d[field_y] = 1 - abs(0.25 - d[field_y]) / 0.25
        if field_y == 'test_y_speed':
            d[field_y] = abs(d[field_y])

        if bool(d["tilt_fall"]):
            x_tilt.append(eval(d[field_x])[0])
            y_tilt.append(d[field_y])
        if bool(d["ground_fall"]):
            x_touch.append(eval(d[field_x])[0])
            y_touch.append(d[field_y])
        else:
            x_good.append(eval(d[field_x])[0])
            y_good.append(d[field_y])

    return x_good, y_good, x_touch, y_touch, x_tilt, y_tilt


def plot_pareto(data):

    plt.figure(figsize=(15, 45), dpi=80)
    gridspec.GridSpec(5, 6)

    # Speed performance
    plt.subplot2grid((5, 6), (0, 0), colspan=3, rowspan=1)
    xg, yg, xto, yto, xti, yti = get_data(data, "physics_init_impedance",
                                          "test_x_speed")
    plt.scatter(xti, yti, marker='^', facecolors='darkred', linewidths=0.01)
    plt.scatter(xto, yto, marker='*', facecolors='royalblue', linewidths=0.01)
    plt.scatter(xg, yg, marker='o', facecolors='olivedrab')
    plt.title("PERFORMANCE: X Speed")
    plt.xlabel('Compliance Gain [log scale]')
    plt.ylabel('Speed [m/s]')
    plt.xscale("log")
    plt.subplot2grid((5, 6), (0, 3), colspan=3, rowspan=1)
    xg, yg, xto, yto, xti, yti = get_data(data, "physics_init_impedance",
                                          "test_x_dist")
    plt.scatter(xti, yti, marker='^', facecolors='darkred', linewidths=0.01)
    plt.scatter(xto, yto, marker='*', facecolors='royalblue', linewidths=0.01)
    plt.scatter(xg, yg, marker='o', facecolors='olivedrab')
    plt.title("PERFORMANCE: X Distance")
    plt.xlabel('Compliance Gain [log scale]')
    plt.ylabel('Distance [m]')
    plt.xscale("log")

    # Energy performance
    plt.subplot2grid((5, 6), (1, 0), colspan=3, rowspan=1)
    xg, yg, xto, yto, xti, yti = get_data(data, "physics_init_impedance",
                                          "test_COT")
    plt.scatter(xti, yti, marker='^', facecolors='darkred', linewidths=0.01)
    plt.scatter(xto, yto, marker='*', facecolors='royalblue', linewidths=0.01)
    plt.scatter(xg, yg, marker='o', facecolors='olivedrab')
    plt.title("PERFORMANCE: Cost of transport")
    plt.xlabel('Compliance Gain [log scale]')
    plt.ylabel('Cost of Transport')
    plt.xscale("log")
    plt.subplot2grid((5, 6), (1, 3), colspan=3, rowspan=1)
    xg, yg, xto, yto, xti, yti = get_data(data, "physics_init_impedance",
                                          "test_power")
    plt.scatter(xti, yti, marker='^', facecolors='darkred', linewidths=0.01)
    plt.scatter(xto, yto, marker='*', facecolors='royalblue', linewidths=0.01)
    plt.scatter(xg, yg, marker='o', facecolors='olivedrab')
    plt.title("PERFORMANCE: Power consumption")
    plt.xlabel('Compliance Gain [log scale]')
    plt.ylabel('Power [W]')
    plt.xscale("log")

    # Deviation
    plt.subplot2grid((5, 6), (2, 0), colspan=3, rowspan=1)
    xg, yg, xto, yto, xti, yti = get_data(data, "physics_init_impedance",
                                          "test_y_speed")
    plt.scatter(xti, yti, marker='^', facecolors='darkred', linewidths=0.01)
    plt.scatter(xto, yto, marker='*', facecolors='royalblue', linewidths=0.01)
    plt.scatter(xg, yg, marker='o', facecolors='olivedrab')
    plt.title("DEVIATION: Y Speed")
    plt.xlabel('Compliance Gain [log scale]')
    plt.ylabel('Y Speed [m/s]')
    plt.xscale("log")
    plt.subplot2grid((5, 6), (2, 3), colspan=3, rowspan=1)
    xg, yg, xto, yto, xti, yti = get_data(data, "physics_init_impedance",
                                          "test_y_dist")
    plt.scatter(xti, yti, marker='^', facecolors='darkred', linewidths=0.01)
    plt.scatter(xto, yto, marker='*', facecolors='royalblue', linewidths=0.01)
    plt.scatter(xg, yg, marker='o', facecolors='olivedrab')
    plt.title("DEVIATION: Y Distance")
    plt.xlabel('Compliance Gain [log scale]')
    plt.ylabel('Y Distance [m]')
    plt.xscale("log")

    # Accuracy and computation
    plt.subplot2grid((5, 6), (3, 0), colspan=3, rowspan=1)
    xg, yg, xto, yto, xti, yti = get_data(data, "physics_init_impedance",
                                          "test_nrmse")
    plt.scatter(xti, yti, marker='^', facecolors='darkred', linewidths=0.01)
    plt.scatter(xto, yto, marker='*', facecolors='royalblue', linewidths=0.01)
    plt.scatter(xg, yg, marker='o', facecolors='olivedrab')
    plt.title("ACCURACY: Normalized RMS Error")
    plt.xlabel('Compliance Gain [log scale]')
    plt.ylabel('NRMSE')
    plt.xscale("log")
    plt.subplot2grid((5, 6), (3, 3), colspan=3, rowspan=1)
    xg, yg, xto, yto, xti, yti = get_data(data, "physics_init_impedance",
                                          "test_average_computation_time")
    plt.scatter(xti, yti, marker='^', facecolors='darkred', linewidths=0.01)
    plt.scatter(xto, yto, marker='*', facecolors='royalblue', linewidths=0.01)
    plt.scatter(xg, yg, marker='o', facecolors='olivedrab')
    plt.title("COMPLEXITY: Average Network Computation Time")
    plt.xlabel('Compliance Gain [log scale]')
    plt.ylabel('Inference Time [s]')
    plt.xscale("log")

    # Stability
    plt.subplot2grid((5, 6), (4, 0), colspan=2, rowspan=1)
    xg, yg, xto, yto, xti, yti = get_data(data, "physics_init_impedance",
                                          "test_z_range")
    plt.scatter(xti, yti, marker='^', facecolors='darkred', linewidths=0.01)
    plt.scatter(xto, yto, marker='*', facecolors='royalblue', linewidths=0.01)
    plt.scatter(xg, yg, marker='o', facecolors='olivedrab')
    plt.title("STABILITY: Height Oscillation range")
    plt.xlabel('Compliance Gain [log scale]')
    plt.ylabel('Z Range [m]')
    plt.xscale("log")
    plt.subplot2grid((5, 6), (4, 2), colspan=2, rowspan=1)
    xg, yg, xto, yto, xti, yti = get_data(data, "physics_init_impedance",
                                          "test_pitch_range")
    plt.scatter(xti, yti, marker='^', facecolors='darkred', linewidths=0.01)
    plt.scatter(xto, yto, marker='*', facecolors='royalblue', linewidths=0.01)
    plt.scatter(xg, yg, marker='o', facecolors='olivedrab')
    plt.title("STABILITY: Pitch Oscillation Range")
    plt.xlabel('Compliance Gain [log scale]')
    plt.ylabel('Pitch Range [rad]')
    plt.xscale("log")
    plt.subplot2grid((5, 6), (4, 4), colspan=2, rowspan=1)
    xg, yg, xto, yto, xti, yti = get_data(data, "physics_init_impedance",
                                          "test_roll_range")
    plt.scatter(xti, yti, marker='^', facecolors='darkred', linewidths=0.01)
    plt.scatter(xto, yto, marker='*', facecolors='royalblue', linewidths=0.01)
    plt.scatter(xg, yg, marker='o', facecolors='olivedrab')
    plt.title("STABILITY: Roll Oscillation Range")
    plt.xlabel('Compliance Gain [log scale]')
    plt.ylabel('Roll Range [rad]')
    plt.xscale("log")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] == "process":
            folder, default_conf = plot_metrics.get_folder()
            data, data_config_fields = plot_metrics.get_data(folder)
            with open(os.path.join(os.path.dirname(folder), "pareto.pkl"), "wb") as f:
                pickle.dump([data, data_config_fields], f, protocol=2)
            exit()

    [data, changing_config] = pickle.load(open(plot_metrics.EXP_FOLDER + "pareto.pkl", "rb"))
    plot_pareto(data)