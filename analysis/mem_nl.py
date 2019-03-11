import os
import numpy as np
import pickle
import sys
from scipy.interpolate import griddata

import matplotlib.colors as cols
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from hyq import analysis

plt.style.use('fivethirtyeight')
plt.rc('text', usetex=False)
plt.rc('font', size=9)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)

MIN_DL = 1
MAX_DL = 120
MIN_ELM = 0
MAX_ELM = 100
PHYS_REG = 0
COMPLIANCE = 75
FOLDER = "/home/gurbain/docker_sim/experiments/mem_nl"


def get_compliance_vals(data):

    compliance_list = []
    for d in data:
        if eval(d['physics_init_impedance'])[0] not in compliance_list:
            compliance_list.append(eval(d['physics_init_impedance'])[0])
    return sorted(compliance_list)


def select_data(data, type_sel="no_joint"):

    new_data = []
    if type_sel == "with_joint":
        for d in data:
            if 'joints' in eval(d['simulation_inputs']):
                new_data.append(d)

    if type_sel == "no_joint":
        for d in data:
            if 'joints' not in eval(d['simulation_inputs']):
                new_data.append(d)

    return new_data


def sort_data(data):

    #data = select_data(data)
    new_data = []
    for d in data:
        # if float(d['physics_noise']) == PHYS_REG:
            #if MIN_DL < int(d['force_delay_line_n']) < MAX_DL and MIN_ELM < int(d['force_elm_n']) < MAX_ELM:
                if eval(d['physics_init_impedance'])[2] == COMPLIANCE:
                    #if not (bool(d["test_fall"]) or bool(d["cl_fall"]) or bool(d["train_fall"])):
                        new_data.append(d)

    print "Sorted data length = " + str(len(new_data))
    if len(new_data) == 0:
        print "Not enough data to print anything! Exiting!"
        exit(0)
    return new_data


def get_mem_nl_vals(data):

    x = []
    y = []
    for d in data:
        dl = int(d['force_delay_line_n'])
        elm = int(d['force_elm_n'])
        if dl not in x:
            x.append(dl)
        if elm not in y:
            y.append(elm)

    return x, y


def get_data(data, field):

    x_set, z_av_tab, z_std_tab, y_set = analysis.get_graph_data(data, 'force_delay_line_n',
                                                                field, 'force_elm_n')
    x = []
    y = []
    z_av = []
    z_std = []
    for i, xs in enumerate(x_set):
        for j, ys in enumerate(y_set):
            x.append(xs)
            y.append(ys)
            if 'x_speed' in field:
                z_av_tab[i, j] = 1 - abs(0.25 - z_av_tab[i, j]) / 0.25
                if z_av_tab[i, j] < 0:
                    z_av_tab[i, j] = 0
            if 'y_speed' in field:
                z_av_tab[i, j] = abs(z_av_tab[i, j])

            z_av.append(z_av_tab[i, j])
            z_std.append(z_std_tab[i, j])

    return x, y, z_av, z_std


def plot_mem_nl(data, phase="test"):

    data = sort_data(data)

    plt.figure(figsize=(12, 45), dpi=80)
    gridspec.GridSpec(4, 6)
    current_cmap = cm.get_cmap()
    current_cmap.set_bad(color='darkred')

    plt.subplot2grid((4, 6), (0, 0), colspan=2, rowspan=1)
    x, y, z_av, z_std = get_data(data, phase + "_x_speed")
    xi = np.arange(MIN_DL, MAX_DL, 0.1)
    yi = np.arange(MIN_ELM, MAX_ELM, 0.1)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z_av, (xi, yi), method='linear')
    plt.imshow(zi, origin='lower',
               extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)],
               interpolation='nearest', aspect='auto')
    # plt.plot(x, y, 'k.')
    plt.colorbar()
    plt.title("PERFORMANCE: Normalized Longitudinal Speed")
    plt.xlabel('Delay Line Size')
    plt.ylabel('ELM Size')
    plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])

    plt.subplot2grid((4, 6), (0, 2), colspan=2, rowspan=1)
    x, y, z_av, z_std = get_data(data, phase + "_y_speed")
    xi = np.arange(MIN_DL, MAX_DL, 0.1)
    yi = np.arange(MIN_ELM, MAX_ELM, 0.1)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z_av, (xi, yi), method='linear')
    plt.imshow(zi, origin='lower',
               extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)],
               interpolation='nearest', aspect='auto')
    # plt.plot(x, y, 'k.')
    plt.colorbar()
    plt.title("DEVIATION: Lateral Speed")
    plt.xlabel('Delay Line Size')
    plt.ylabel('ELM Size')
    plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])

    plt.subplot2grid((4, 6), (0, 4), colspan=2, rowspan=1)
    x, y, z_av, z_std = get_data(data, phase + "_grf_max")
    xi = np.arange(MIN_DL, MAX_DL, 0.1)
    yi = np.arange(MIN_ELM, MAX_ELM, 0.1)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z_av, (xi, yi), method='linear')
    plt.imshow(zi, origin='lower',
               extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)],
               interpolation='nearest', aspect='auto',
               norm=cols.LogNorm(vmin=max(0.0001, min(z_av)), vmax=max(z_av)))
    # plt.plot(x, y, 'k.')
    #plt.colorbar()
    plt.title("STABILITY: Maximal Ground Force Reaction")
    plt.xlabel('Delay Line Size')
    plt.ylabel('ELM Size')
    plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])

    plt.subplot2grid((4, 6), (1, 0), colspan=3, rowspan=1)
    x, y, z_av, z_std = get_data(data, phase + "_COT")
    xi = np.arange(MIN_DL, MAX_DL, 0.1)
    yi = np.arange(MIN_ELM, MAX_ELM, 0.1)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z_av, (xi, yi), method='linear')
    plt.imshow(zi, origin='lower',
               extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)],
               interpolation='nearest', aspect='auto',
               norm=cols.LogNorm(vmin=max(0.0001, min(z_av)), vmax=max(z_av)))
    # plt.plot(x, y, 'k.')
    #plt.colorbar()
    plt.title("PERFORMANCE: Cost of Transport")
    plt.xlabel('Delay Line Size')
    plt.ylabel('ELM Size')
    plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])

    plt.subplot2grid((4, 6), (1, 3), colspan=3, rowspan=1)
    x, y, z_av, z_std = get_data(data, phase + "_power")
    xi = np.arange(MIN_DL, MAX_DL, 0.1)
    yi = np.arange(MIN_ELM, MAX_ELM, 0.1)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z_av, (xi, yi), method='linear')
    plt.imshow(zi, origin='lower',
               extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)],
               interpolation='nearest', aspect='auto',
               norm=cols.LogNorm(vmin=max(0.0001, min(z_av)), vmax=max(z_av)))
    # plt.plot(x, y, 'k.')
    #plt.colorbar()
    plt.title("PERFORMANCE: Power Consumption")
    plt.xlabel('Delay Line Size')
    plt.ylabel('ELM Size')
    plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])

    plt.subplot2grid((4, 6), (2, 0), colspan=3, rowspan=1)
    x, y, z_av, z_std = get_data(data, phase + "_nrmse")
    xi = np.arange(MIN_DL, MAX_DL, 0.1)
    yi = np.arange(MIN_ELM, MAX_ELM, 0.1)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z_av, (xi, yi), method='linear')
    plt.imshow(zi, origin='lower',
               extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)],
               interpolation='nearest', aspect='auto',
               norm=cols.LogNorm(vmin=max(0.0001, min(z_av)), vmax=max(z_av)))
    # plt.plot(x, y, 'k.')
    #plt.colorbar()
    plt.title("ACCURACY: Normalized RMS Error")
    plt.xlabel('Delay Line Size')
    plt.ylabel('ELM Size')
    plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])

    plt.subplot2grid((4, 6), (2, 3), colspan=3, rowspan=1)
    x, y, z_av, z_std = get_data(data, "train_average_computation_time")
    xi = np.arange(MIN_DL, MAX_DL, 0.1)
    yi = np.arange(MIN_ELM, MAX_ELM, 0.1)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z_av, (xi, yi), method='linear')
    plt.imshow(zi, origin='lower',
               extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)],
               interpolation='nearest', aspect='auto')
    # plt.plot(x, y, 'k.')
    plt.colorbar()
    plt.title("COMPLEXITY: Average Network Computation Time")
    plt.xlabel('Delay Line Size')
    plt.ylabel('ELM Size')
    plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])

    plt.subplot2grid((4, 6), (3, 0), colspan=2, rowspan=1)
    x, y, z_av, z_std = get_data(data, phase + "_z_range")
    xi = np.arange(MIN_DL, MAX_DL, 0.1)
    yi = np.arange(MIN_ELM, MAX_ELM, 0.1)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z_av, (xi, yi), method='linear')
    plt.imshow(zi, origin='lower',
               extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)],
               interpolation='nearest', aspect='auto')
    # plt.plot(x, y, 'k.')
    plt.colorbar()
    plt.title("STABILITY: Height Oscillation range")
    plt.xlabel('Delay Line Size')
    plt.ylabel('ELM Size')
    plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])

    plt.subplot2grid((4, 6), (3, 2), colspan=2, rowspan=1)
    x, y, z_av, z_std = get_data(data, phase + "_pitch_range")
    xi = np.arange(MIN_DL, MAX_DL, 0.1)
    yi = np.arange(MIN_ELM, MAX_ELM, 0.1)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z_av, (xi, yi), method='linear')
    plt.imshow(zi, origin='lower',
               extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)],
               interpolation='nearest', aspect='auto')
    # plt.plot(x, y, 'k.')
    plt.colorbar()
    plt.title("STABILITY: Pitch Oscillation Range")
    plt.xlabel('Delay Line Size')
    plt.ylabel('ELM Size')
    plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])

    plt.subplot2grid((4, 6), (3, 4), colspan=2, rowspan=1)
    x, y, z_av, z_std = get_data(data, phase + "_roll_range")
    xi = np.arange(MIN_DL, MAX_DL, 0.1)
    yi = np.arange(MIN_ELM, MAX_ELM, 0.1)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z_av, (xi, yi), method='linear')
    plt.imshow(zi, origin='lower',
               extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)],
               interpolation='nearest', aspect='auto')
    # plt.plot(x, y, 'k.')
    plt.colorbar()
    plt.title("STABILITY: Roll Oscillation Range")
    plt.xlabel('Delay Line Size')
    plt.ylabel('ELM Size')
    plt.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1] == "process":
            data, data_config_fields = analysis.get_data(FOLDER)
            with open(os.path.join(FOLDER, "mem_nl.pkl"), "wb") as f:
                pickle.dump([data, data_config_fields], f, protocol=2)
            exit()

    [data, changing_config] = pickle.load(open(os.path.join(FOLDER, "mem_nl.pkl"), "rb"))
    print changing_config
    plot_mem_nl(data)