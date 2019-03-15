#!/usr/bin/env python

import datetime
import numpy as np
import os
import pause
import pickle
import rospy as ros
from scipy.interpolate import interp1d
import sys
import time

from PyQt5 import QtCore, QtWidgets, QtGui
import qdarkstyle

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import physics
import utils
import model

plt.style.use('fivethirtyeight')
plt.rc('font', family='Bitstream Vera Sans')
plt.rc('axes', facecolor='white')
plt.rc('figure', autolayout=True)
plt.rc('xtick', color='white')
plt.rc('ytick', color='white')

# plt.style.use('fivethirtyeight')
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rc('axes', facecolor='white')
# plt.rc('savefig', facecolor='white')
# plt.rc('figure', autolayout=True)

RESULT_FOLDER = '/home/gabs48/src/quadruped/tigrillo2/data/optim_calib/results'


class SimpleSimulation(object):

    def __init__(self, win, sim_id=0, time_step=0.02):

        self.physics = physics.Gazebo("tigrillo.world", view=True)
        self.sim_id = sim_id
        self.win = win
        self.time_step = time_step

        print sim_id

        ros.init_node('view', anonymous=True)
        ros.on_shutdown(utils.cleanup)

        # Simulation data retrieved from file
        data = self.win.sel_conf[sim_id]
        data_init = self.win.sel_conf[0]
        self.f_fl = interp1d(data_init["t_act"] , data_init["fl_act"], assume_sorted=False, fill_value="extrapolate")
        self.f_fr = interp1d(data_init["t_act"] , data_init["fr_act"], assume_sorted=False, fill_value="extrapolate")
        self.f_bl = interp1d(data_init["t_act"] , data_init["bl_act"], assume_sorted=False, fill_value="extrapolate")
        self.f_br = interp1d(data_init["t_act"] , data_init["br_act"], assume_sorted=False, fill_value="extrapolate")
        self.params_unormed = utils.unorm(data["params"], data_init["params_min"], data_init["params_max"])
        self.config = data_init["config"]
        self.file = "/home/gabs48/.gazebo/models/tigrillo/model.sdf"#data_init["model_file"]
        self.time_bias = data_init["start_time"]
        self.sim_time = data_init["stop_time"] - data_init["start_time"]

    def simulate(self):

        # Create the model
        fg = model.SDFileGenerator(self.config, self.file, model_scale=1, gazebo=True)
        fg.generate()

        # Create the simulator
        self.physics.start()
        while not self.physics.is_sim_started():
            time.sleep(0.001)

        # Perform simulation loop
        rt_init = datetime.datetime.now()
        st = 0
        i = 0
        while st < self.sim_time:

            # Actuate
            st = self.physics.get_gazebo_time()
            rt = rt_init + datetime.timedelta(seconds=((i+1) * self.time_step))
            self.physics.actuate(self.getActuator(st + self.time_bias))

            # Pause
            pause.until(rt)
            i += 1
        
        # Stop the simulator
        self.physics.stop()
        return 0

    def getActuator(self, st):

        fl = self.f_fl(st) #* self.params_unormed[12] + self.params_unormed[13]
        fr = self.f_fr(st) #* self.params_unormed[12] + self.params_unormed[13]
        bl = self.f_bl(st) #* self.params_unormed[12] + self.params_unormed[14]
        br = self.f_br(st) #* self.params_unormed[12] + self.params_unormed[14]

        return [fl, fr, bl, br]


class SimpleTable(QtWidgets.QTableWidget):

    def __init__(self, data, *args):

        QtWidgets.QTableWidget.__init__(self, *args)
        self.data = data
        self.fillData()
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

        # Add CTRL-C action
        #self.copy = CopySelectedCellsAction(self)

    def fillData(self):

        headers = ['Parameter', 'Value']
        i = 0
        for key in sorted(self.data):
            name_item = QtWidgets.QTableWidgetItem(key.replace('_', ' ').title())
            self.setItem(i, 0, name_item)
            value_item = QtWidgets.QTableWidgetItem(str(self.data[key]))
            if key == 'iter':
                value_item = QtWidgets.QTableWidgetItem(str(self.data[key] + 1))
            if key == "params":
                value_item = QtWidgets.QTableWidgetItem(str(self.data[key].tolist()))
            self.setItem(i, 1, value_item)
            i += 1

        self.setHorizontalHeaderLabels(headers)


class SimpleFigure(FigureCanvas):

    def __init__(self, parent=None, subplot=111, width=8, height=6, dpi=100):

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor("None")
        self.axes = self.fig.add_subplot(subplot)

        FigureCanvas.__init__(self, self.fig)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setStyleSheet("background-color:transparent;")
        self.updateGeometry()

    def save(self, name="figure.png"):

        self.axes.xaxis.label.set_color('black')
        self.axes.yaxis.label.set_color('black')
        self.axes.title.set_color('black')
        self.axes.tick_params(axis='x', colors='black')
        self.axes.tick_params(axis='y', colors='black')
        self.fig.set_size_inches(7, 5)
        self.fig.savefig(name, format='png', dpi=300)
        self.axes.xaxis.label.set_color('white')
        self.axes.yaxis.label.set_color('white')
        self.axes.title.set_color('white')
        self.axes.tick_params(axis='x', colors='white')
        self.axes.tick_params(axis='y', colors='white')


class VizWin(QtWidgets.QGridLayout):

    def __init__(self, win):

        QtWidgets.QGridLayout.__init__(self)

        self.setGeometry(QtCore.QRect(0, 0, 1000, 600))

        self.win = win

    def plotFigure(self, name):

        if name in ["Plot FL", "Plot FR", "Plot BR", "Plot BL"]:
            self.plotSensor(name)
        if name  == "CMA Evolution":
            self.plotEvolution()
        if name  == "Sim Parameters":
            self.showSimParams()
        if name  == "Optim Parameters":
            self.showOptimParams()
        if name  == "Simulate Best":
            self.simulate(current=False)
        if name  == "Simulate":
            self.simulate(current=True)
        if name == "Parameters Evolution":
            self.plotParams()

    def plotSensor(self, name):

        self.clean()
        data_init = self.win.sel_conf[0]
        data = self.win.sel_conf[self.win.sel_ite]
        l = name.split()[-1]

        # Multi lines
        if len(data_init[l.lower() + "_rob"].shape) > 1:

            self.plot = SimpleFigure(subplot=211)
            self.addWidget(self.plot)
            self.plot.axes.cla()
            rob_mean, sim_mean = utils.center_norm_2(np.mean(data_init[l.lower() + "_rob"], axis=1),
                                                     np.mean(data[l.lower() + "_sim"], axis=1))
            self.plot.axes.plot(data_init["t"], data_init[l.lower() + "_rob"], linewidth=0.4, color="skyblue")
            self.plot.axes.plot(data_init["t"], data[l.lower() + "_sim"], linewidth=0.8, color="orchid")
            self.plot.axes.set_xlim([data_init["t"][0], data_init["t"][-1]])

            self.plot.axes2 = self.plot.fig.add_subplot(212)
            self.plot.axes2.plot(data_init["t"], rob_mean, linewidth=2.5, color=self.getStyleColors()[0], label="Averaged Centered " + l + " Rob")
            self.plot.axes2.plot(data_init["t"], sim_mean, linewidth=2.5, color=self.getStyleColors()[1], label="Averaged Centered " + l + " Sim")
            self.plot.axes2.fill_between(data_init["t"], np.abs(sim_mean-rob_mean), alpha=0.3, edgecolor=self.getStyleColors()[2], facecolor=self.getStyleColors()[2], label="Absolute Error")
            self.plot.axes2.set_xlim([data_init["t"][0], data_init["t"][-1]])
            self.plot.axes.legend(loc='best', fontsize="x-small")
            # self.plot.axes.set_title("Periodic average of the robot under-actuated front left leg sensor", fontsize=14)
            # self.plot.axes.set_ylabel('Leg knee angle (degrees)')
            # self.plot.axes.set_xlabel('Time (s)')

        else:
            self.plot = SimpleFigure()
            self.addWidget(self.plot)
            self.plot.axes.cla()
            self.plot.axes.plot(data_init["t"], data_init[l.lower() + "_rob"], linewidth=1, label=l + " Leg Robot")
            self.plot.axes.plot(data_init["t"], data[l.lower() + "_sim"], linewidth=1, label=l + " Leg Simulation")
            self.plot.axes.legend()

        self.plot.draw()
        # self.plot.save("average.png")

    def plotEvolution(self):

        self.clean()
        scores = [t["score"] for t in self.win.sel_conf]
        pop_size = self.win.sel_conf[0]["pop"]
        try:
            x, y_min, y_max, y_av = self.rearrangePop(scores, pop_size)
        except AssertionError as e:
            print e
            return

        self.plot = SimpleFigure()
        self.addWidget(self.plot)

        self.plot.axes.cla()
        self.plot.axes.plot(x, y_max, linestyle="-", color=self.getStyleColors()[1], linewidth=1, label="Generation Maximum")
        self.plot.axes.plot(x, y_av, linestyle="-", color=self.getStyleColors()[3], linewidth=1, label="Generation Average")
        self.plot.axes.plot(x, y_min, linestyle="-", color=self.getStyleColors()[0], linewidth=1, label="Generation Minimum")
        self.plot.axes.set_title("Training score of CMA-ES algorithm with popSize = " + str(pop_size), fontsize=14)
        self.plot.axes.set_ylabel('Sensor Error')
        self.plot.axes.set_xlabel('Generation Epoch')
        self.plot.axes.legend(loc="upper right", fontsize="small")
        self.plot.axes.xaxis.label.set_color('white')
        self.plot.axes.yaxis.label.set_color('white')
        self.plot.axes.title.set_color('white')
        # self.plot.save("cma_results.png")

    def plotParams(self, n=4):

        self.clean()
        
        t = range(0, len(self.win.sel_conf))
        params = np.array([i["params"] for i in self.win.sel_conf])[:, -n:].T
        labels = np.array(self.win.sel_conf[0]["params_names"])[-n:]

        self.plot = SimpleFigure()
        self.addWidget(self.plot)

        for y_arr, label in zip(params, labels):
            self.plot.axes.plot(t, y_arr, linewidth=0.8, label=label)
        
        self.plot.axes.set_title("Evolution of Optimized Parameters", fontsize=14)
        self.plot.axes.set_ylabel('Normalized parameters')
        self.plot.axes.set_xlabel('Generation Epoch')
        self.plot.axes.legend(loc="upper left", fontsize="small")
        self.plot.axes.xaxis.label.set_color('white')
        self.plot.axes.yaxis.label.set_color('white')
        self.plot.axes.title.set_color('white')
        # self.plot.save("cma_results.png")

    def showSimParams(self):

        self.clean()

        params_unormed = utils.unorm(self.win.sel_conf[self.win.sel_ite]["params"], 
                                      self.win.sel_conf[0]["params_min"],
                                      self.win.sel_conf[0]["params_max"])
        params_names = self.win.sel_conf[0]["params_names"]
        params_units = self.win.sel_conf[0]["params_units"]
        data = {k:self.win.sel_conf[self.win.sel_ite][k] 
                for k in ('iter', "score", "params", "elapsed time") if k in self.win.sel_conf[self.win.sel_ite]}

        for i, p in enumerate(params_unormed):
            data[params_names[i]] = "{0:.4f} ".format(p) + str(params_units[i])

        self.table = SimpleTable(data, len(data), 2)
        self.addWidget(self.table)

        return

    def showOptimParams(self):

        self.clean()

        scores = [self.win.sel_conf[i]["score"] for i in range(len(self.win.sel_conf))]
        data = {k:self.win.sel_conf[0][k] for k in ('bag_file', "sim_file", "sim_time", "start_time", "stop_time",
                                                    'eval_points', 'start_eval_time', 'stop_eval_time', 'pool_number', 
                                                    'init_var', 'min', 'max', 'pop', 'max_iter', 'score_method',
                                                    'params_min', 'params_max', 'params_names', 'params_units')
                                          if k in self.win.sel_conf[0]}
        data["best_score"] = min(scores)
        data["best_iteration"] = scores.index(min(scores)) + 1
        self.table = SimpleTable(data, len(data), 2)
        self.addWidget(self.table)

        return

    def simulate(self, current=True):

        if current:
            sim_id = self.win.sel_ite
        else:
            scores = [self.win.sel_conf[i]["score"] for i in range(len(self.win.sel_conf))]
            sim_id = scores.index(min(scores))
        s = SimpleSimulation(win=self.win, sim_id=sim_id, time_step=0.02)
        s.simulate()
        del s

    def clean(self):

        for i in reversed(range(self.count())): 
            self.itemAt(i).widget().setParent(None)

    def getStyleColors(self):

        if 'axes.prop_cycle' in plt.rcParams:
            cols = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
        else:
            cols = ['b', 'r', 'y', 'g', 'k']
        return cols

    def rearrangePop(self, scores, ps):

        array = np.array(scores)
        array = array[0:len(scores) -(len(scores) % ps)]
        # assert len(scores) % ps == 0, "The total number of iteration (" + str(len(scores)) + \
        #                                ") shall be a multiple of the population size (" + \
        #                                str(ps) + "). Please verify the file " +  \
        #                                self.win.sel_exp + " or this sript!"

        matrix = np.reshape(array, (-1, ps))
        y_min = np.min(matrix, axis=1)
        y_max = np.max(matrix, axis=1)
        y_av = np.mean(matrix, axis=1)
        x = np.array(range((y_av.size))) + 1
        return x, y_min, y_max, y_av


class IteListWin(QtWidgets.QGridLayout):

    def __init__(self, win):

        QtWidgets.QGridLayout.__init__(self)
        self.setGeometry(QtCore.QRect(0, 0, 150, 400))

        self.label = QtWidgets.QLabel()
        self.label.setText("Epoch List")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.addWidget(self.label)

        self.list = QtWidgets.QListWidget()
        self.list.setMinimumWidth(100)
        self.list.setMaximumWidth(350)
        self.addWidget(self.list)

        self.win = win

        self.list.itemClicked.connect(self.selectIteration)
        self.list.currentItemChanged.connect(self.selectIteration)

    def loadIterations(self):

        self.list.clear()

        if os.path.isfile(self.win.sel_exp + "/cmaes_evolution.pkl"):
            self.win.sel_conf = pickle.load(open(self.win.sel_exp + "/cmaes_evolution.pkl", "rb"))
            for i in range(len(self.win.sel_conf)):
                item = QtWidgets.QListWidgetItem()
                item.setText("Iteration " + str(i+1))
                item.setData(QtCore.Qt.UserRole, i)
                self.list.addItem(item)

            self.win.sel_ite = 0

        else:
            item = QtWidgets.QListWidgetItem()
            item.setText("Experiment produced no data")
            item.setData(QtCore.Qt.UserRole, None)
            self.list.addItem(item)
            self.win.sel_ite = None
            self.sel_conf = None

    def selectIteration(self, item):

        if "data" in dir(item):
            self.win.sel_ite = item.data(QtCore.Qt.UserRole)
        if self.win.last_action != None:
            self.win.exp_butt_lay.dispatchAction(self.win.last_action)


class ExpListWin(QtWidgets.QGridLayout):

    def __init__(self, win):

        QtWidgets.QGridLayout.__init__(self)
        self.setGeometry(QtCore.QRect(0, 0, 150, 400))

        self.label = QtWidgets.QLabel()
        self.label.setText("Experiment List")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.addWidget(self.label)

        self.list = QtWidgets.QListWidget()
        self.list.setMinimumWidth(100)
        self.list.setMaximumWidth(350)
        self.addWidget(self.list)

        self.win = win

        self.loadExperiments()
        self.list.itemClicked.connect(self.selectExperiment)
        self.list.currentItemChanged.connect(self.selectExperiment)

    def loadExperiments(self):

        for dirname, dirnames, filenames in os.walk(self.win.folder):

            dirnames.sort(reverse=True)
            for subdirname in dirnames:
                date = datetime.datetime.strptime(subdirname, '%Y%m%d-%H%M%S')
                item = QtWidgets.QListWidgetItem()
                item.setText(date.strftime("Exp %d/%m/%Y - %H:%M:%S"))
                item.setData(QtCore.Qt.UserRole, dirname + "/" + subdirname)
                self.list.addItem(item)

    def selectExperiment(self, item):

        self.win.sel_exp = item.data(QtCore.Qt.UserRole)
        self.win.ite_list_lay.loadIterations()


class ExpButWin(QtWidgets.QGridLayout):

    def __init__(self, win):

        QtWidgets.QGridLayout.__init__(self)
        self.setGeometry(QtCore.QRect(0, 0, 1000, 100))

        self.win = win

        self.addLegend()
        self.addButtons()

    def addButtons(self):
        
        self.b1 = QtWidgets.QPushButton("CMA Evolution")
        self.b1.installEventFilter(self)
        self.addWidget(self.b1, 1, 0)

        self.b2 = QtWidgets.QPushButton("Parameters Evolution")
        self.b2.installEventFilter(self)
        self.addWidget(self.b2, 2, 0)

        self.b3 = QtWidgets.QPushButton("Simulate Best")
        self.b3.installEventFilter(self)
        self.addWidget(self.b3, 1, 1)

        self.b4 = QtWidgets.QPushButton("Optim Parameters")
        self.b4.installEventFilter(self)
        self.addWidget(self.b4, 2, 1)

    def addLegend(self):
        
        self.l1 = QtWidgets.QLabel()
        self.l1.setText("Optimization Functions")
        self.l1.setAlignment(QtCore.Qt.AlignCenter)
        self.addWidget(self.l1, 0, 0, 1, 2)

    def eventFilter(self, object, event):

        if event.type() == QtCore.QEvent.MouseButtonPress:
            return self.dispatchAction(object.text())

        elif event.type() == QtCore.QEvent.HoverMove:
            return self.displayHelp(object.text())

        return False

    def dispatchAction(self, action):

        if self.win.sel_conf:
            self.win.viz_lay.plotFigure(action)
            self.win.last_action = action
            return True
        else:
            self.win.displayStatus("Please select experiment and iteration before using this function", 3000)

    def displayHelp(self, action):

            if action == "Simulate Best":
                self.win.displayStatus("Replay the simulation of the best individual in the experiment")
            elif action == "Optim Parameters":
                self.win.displayStatus("Display the parameters of the optimization process")
            elif action == "CMA Evolution":
                self.win.displayStatus("Display the optimization evolution across generations")
            elif action == "Parameters Evolution":
                self.win.displayStatus("Display the parameters convergence across generations")
            return True


class IteButWin(QtWidgets.QGridLayout):

    def __init__(self, win):

        QtWidgets.QGridLayout.__init__(self)
        self.setGeometry(QtCore.QRect(0, 0, 1000, 100))

        self.win = win

        self.addLegend()
        self.addButtons()

    def addButtons(self):
        
        self.b1 = QtWidgets.QPushButton("Plot FL")
        self.b1.installEventFilter(self)
        self.addWidget(self.b1, 1, 0)

        self.b2 = QtWidgets.QPushButton("Plot FR")
        self.b2.installEventFilter(self)
        self.addWidget(self.b2, 1, 1)

        self.b3 = QtWidgets.QPushButton("Simulate")
        self.b3.installEventFilter(self)
        self.addWidget(self.b3, 1, 2)

        self.b4 = QtWidgets.QPushButton("Plot BL")
        self.b4.installEventFilter(self)
        self.addWidget(self.b4, 2, 0)

        self.b5 = QtWidgets.QPushButton("Plot BR")
        self.b5.installEventFilter(self)
        self.addWidget(self.b5, 2, 1)

        self.b6 = QtWidgets.QPushButton("Spectogram")
        self.b6.installEventFilter(self)
        self.addWidget(self.b6, 2, 2)

        self.b7 = QtWidgets.QPushButton("Sim Parameters")
        self.b7.installEventFilter(self)
        self.addWidget(self.b7, 1, 3)

    def addLegend(self):

        self.l1 = QtWidgets.QLabel()
        self.l1.setText("Iteration Functions")
        self.l1.setAlignment(QtCore.Qt.AlignCenter)
        self.addWidget(self.l1, 0, 0, 1, 4)

    def eventFilter(self, object, event):

        if event.type() == QtCore.QEvent.MouseButtonPress:
            return self.dispatchAction(object.text())

        elif event.type() == QtCore.QEvent.HoverMove:
            return self.displayHelp(object.text())

        return False

    def dispatchAction(self, action):

        if self.win.sel_conf and (self.win.sel_ite is not None):
            self.win.viz_lay.plotFigure(action)
            self.win.last_action = action
            return True
        else:
            self.win.viz_lay.clean()
            self.win.displayStatus("Please select experiment and iteration before using this function", 3000)

    def displayHelp(self, action):

        if "Plot" in action:
            self.win.displayStatus("Plot the sensor signals on robot and in simulation")
        elif action == "Optim Parameters":
            self.win.displayStatus("Display the parameters of the specific iteration")
        elif action == "Spectogram":
            self.win.displayStatus("Plot the Spectogram of the signal difference between robot and simulation")
        return True


class AppWin(QtWidgets.QMainWindow):

    def __init__(self, folder):

        # Init
        QtWidgets.QMainWindow.__init__(self)

        self.folder = folder
        self.sel_exp = None
        self.sel_ite = None
        self.sel_conf = None
        self.last_action = None

        self.initUI()

    def initUI(self):

        self.resize(1200, 800)
        
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Tigrillo Optimization Viewer")
        self.displayStatus("This software allows to browse experiment folder and display results", 4000)
        self.setWindowIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_ComputerIcon')))

        self.constructUI()
        self.moveUI()
        self.show()

    def constructUI(self):

        # Create top menu and shortcuts
        #self.file_menu = QtWidgets.QMenu('&File', self)
        #self.file_menu.addAction('&Quit', self.quitUI, QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        #self.menuBar().addMenu(self.file_menu)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+D"), self, self.quitUI)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+W"), self, self.quitUI)

        # Create frame structure
        self.main_window = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        win_width = self.frameGeometry().width()
        win_height = self.frameGeometry().height()

        self.main_sel_pan = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.main_vis_pan = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        self.ite_list_lay = IteListWin(self)
        self.exp_list_lay = ExpListWin(self)
        self.ite_list = QtWidgets.QWidget()
        self.exp_list = QtWidgets.QWidget()

        self.exp_list.setLayout(self.exp_list_lay)
        self.main_sel_pan.addWidget(self.exp_list)
        self.ite_list.setLayout(self.ite_list_lay)
        self.main_sel_pan.addWidget(self.ite_list)

        self.exp_butt_lay = ExpButWin(self)
        self.ite_butt_lay = IteButWin(self)
        self.viz_lay = VizWin(self)
        self.exp_butt = QtWidgets.QWidget()
        self.ite_butt = QtWidgets.QWidget()
        self.viz = QtWidgets.QWidget()

        self.exp_butt.setLayout(self.exp_butt_lay)
        self.main_vis_pan.addWidget(self.exp_butt)
        self.viz.setLayout(self.viz_lay)
        self.main_vis_pan.addWidget(self.viz)
        self.ite_butt.setLayout(self.ite_butt_lay)
        self.main_vis_pan.addWidget(self.ite_butt)

        self.main_window.addWidget(self.main_sel_pan)
        self.main_window.addWidget(self.main_vis_pan)

        # Set focus
        self.main_window.setFocus()
        self.setCentralWidget(self.main_window)

    def moveUI(self):

        frameGm = self.frameGeometry()
        screen = QtWidgets.QApplication.desktop().screenNumber(QtWidgets.QApplication.desktop().cursor().pos())
        centerPoint = QtWidgets.QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())

    def resizeEvent(self, event):

        win_width = self.frameGeometry().width()
        win_height = self.frameGeometry().height()

        self.ite_list.setMinimumWidth(win_width/5)
        self.ite_list.setMaximumWidth(win_width/3)
        self.exp_list.setMinimumWidth(win_width/5)
        self.exp_list.setMaximumWidth(win_width/3)
        self.exp_butt.setMinimumHeight(win_height/10)
        self.ite_butt.setMinimumHeight(win_height/10)
        self.viz.setMinimumHeight(win_height/2)
        self.exp_butt.setMaximumHeight(win_height/7)
        self.ite_butt.setMaximumHeight(win_height/7)
        self.viz.setMaximumHeight(9*win_height/10)

        QtWidgets.QMainWindow.resizeEvent(self, event)

    def quitUI(self):

        self.close()

    def closeEvent(self, ce):

        utils.cleanup()
        self.quitUI()

    def displayStatus(self, msg, t=1000):

        self.statusBar().showMessage(msg, t)


def gui():

    app = QtWidgets.QApplication(sys.argv)
    win = AppWin(RESULT_FOLDER)

    # Dark style
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    sys.exit(app.exec_())


if __name__ == '__main__':

     gui()
