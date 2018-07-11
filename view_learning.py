#!/usr/bin/env python

import datetime
from hyperopt import Trials
import numpy as np
import os
import pickle
import sys
import time

from keras.callbacks import History

from PyQt5 import QtCore, QtWidgets, QtGui
import qdarkstyle

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

import network
import utils

plt.style.use('fivethirtyeight')
plt.rc('font', family='Bitstream Vera Sans')
plt.rc('axes', facecolor='white')
plt.rc('figure', autolayout=True)
plt.rc('xtick', color='white')
plt.rc('ytick', color='white')


RESULT_FOLDER = '/home/gurbain/hyq_ml/data/hypopt'


class SimpleTable(QtWidgets.QTableWidget):

    def __init__(self, data, *args):

        QtWidgets.QTableWidget.__init__(self, *args)
        self.data = data
        self.fillData()
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def fillData(self):

        headers = ['Parameter', 'Value']
        i = 0
        for key in sorted(self.data):
            name_item = QtWidgets.QTableWidgetItem(key.replace('_', ' ').title())
            self.setItem(i, 0, name_item)
            value_item = QtWidgets.QTableWidgetItem(str(self.data[key]))
            if key == 'iter':
                value_item = QtWidgets.QTableWidgetItem(str(self.data[key] + 1))
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

        if name in ["Loss", "Accuracy"]:
            self.plotMetrics(name)
        if name  == "Layer Size":
            self.plotLossLayers(name)
        if name  == "Layer Number":
            self.plotLossLayers(name)
        if name  == "Parameter Search Evol":
            self.plotHypOptEvo()
        if name  == "Prediction":
            self.plotPrediction()

    def plotMetrics(self, name):

        nn = network.FeedForwardNN()
        nn.verbose = 0
        nn.load(self.win.sel_exp, self.win.sel_ite)
        hist = nn.history

        self.clean()
        self.plot = SimpleFigure()
        self.addWidget(self.plot)
        self.addWidget(NavigationToolbar(self.plot, self.win))
        self.plot.axes.cla()

        if name == "Accuracy":
            self.plot.axes.plot(hist.history["acc"], linewidth=1, label="Training")
            self.plot.axes.plot(hist.history["val_acc"], linewidth=1, label="Validation")
        if name == "Loss":
            self.plot.axes.semilogy(hist.history["loss"], linewidth=1, label="Training")
            self.plot.axes.semilogy(hist.history["val_loss"], linewidth=1, label="Validation")
        self.plot.axes.set_title('Model accuracy', fontsize=14)
        self.plot.axes.set_ylabel('Accuracy')
        self.plot.axes.set_xlabel('Epoch')
        self.plot.axes.legend(loc='upper left')
        self.plot.axes.xaxis.label.set_color('white')
        self.plot.axes.yaxis.label.set_color('white')
        self.plot.axes.title.set_color('white')

        self.plot.draw()

    def plotPrediction(self):

        nn = network.FeedForwardNN()
        nn.verbose = 0
        nn.load(self.win.sel_exp, self.win.sel_ite)
        y_t, y_p, s = nn.evaluate(False)

        self.clean()
        self.plot = SimpleFigure()
        self.addWidget(self.plot)
        self.addWidget(NavigationToolbar(self.plot, self.win))

        self.plot.axes.cla()
        self.plot.axes.plot(y_t[:, 0], linewidth=1, label="Real")
        self.plot.axes.plot(y_p[:, 0], linewidth=1, label="Predicted")
        self.plot.axes.plot(np.abs(y_t - y_p)[:, 0], linewidth=1,
                            label="MAE error")
        self.plot.axes.set_title('Predicted Signal', fontsize=14)
        self.plot.axes.set_ylabel('First Joint Position')
        self.plot.axes.set_xlabel('Time')
        self.plot.axes.legend(loc='upper left')
        self.plot.axes.xaxis.label.set_color('white')
        self.plot.axes.yaxis.label.set_color('white')
        self.plot.axes.title.set_color('white')

        self.plot.draw()

    def plotLossLayers(self, name):

        self.clean()

        if name == "Layer Size":
            n_layers = [r["params"]["s_l"] for r in self.win.sel_conf.results]
            margin = 10
            label = "Size of Layers"
        if name == "Layer Number":
            n_layers = [r["params"]["n_l"] for r in self.win.sel_conf.results]
            margin = 1
            label = "Number of Layers"
        losses = [r["loss"] for r in self.win.sel_conf.results]

        self.plot = SimpleFigure()
        self.addWidget(self.plot)
        self.addWidget(NavigationToolbar(self.plot, self.win))

        self.plot.axes.cla()
        self.plot.axes.plot(n_layers, losses, linestyle='None', marker='.')
        self.plot.axes.set_title("Test Losses for different NN Architectures",
                                 fontsize=14)

        self.plot.axes.set_xlim([min(n_layers) - margin,
                                 max(n_layers) + margin])
        self.plot.axes.set_ylim([min(losses) - 0.0001, max(losses) + 0.0001])
        self.plot.axes.set_ylabel('Loss')
        self.plot.axes.set_xlabel(label)
        self.plot.axes.xaxis.label.set_color('white')
        self.plot.axes.yaxis.label.set_color('white')
        self.plot.axes.title.set_color('white')

    def plotHypOptEvo(self):

        self.clean()

        iteration = range(len(self.win.sel_conf.results))
        losses = [r["loss"] for r in self.win.sel_conf.results]

        self.plot = SimpleFigure()
        self.addWidget(self.plot)
        self.addWidget(NavigationToolbar(self.plot, self.win))

        self.plot.axes.cla()
        self.plot.axes.plot(iteration, losses, linestyle='None', marker='.')
        self.plot.axes.set_title("Evolution of the Hyper Parameters Search",
                                 fontsize=14)

        self.plot.axes.set_xlim([min(iteration) - 1, max(iteration) + 1])
        self.plot.axes.set_ylabel('Loss')
        self.plot.axes.set_xlabel('Search Epoch')
        self.plot.axes.xaxis.label.set_color('white')
        self.plot.axes.yaxis.label.set_color('white')
        self.plot.axes.title.set_color('white')

    def clean(self):

        for i in reversed(range(self.count())):
            self.itemAt(i).widget().setParent(None)

    def getStyleColors(self):

        if 'axes.prop_cycle' in plt.rcParams:
            cols = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
        else:
            cols = ['b', 'r', 'y', 'g', 'k']
        return cols


class IteListWin(QtWidgets.QGridLayout):

    def __init__(self, win):

        QtWidgets.QGridLayout.__init__(self)
        self.setGeometry(QtCore.QRect(0, 0, 150, 400))

        self.label = QtWidgets.QLabel()
        self.label.setText("Metaparameters List")
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

        if os.path.isfile(self.win.sel_exp + "/hyperopt.pkl"):
            self.win.sel_conf =  pickle.load(open(self.win.sel_exp + "/hyperopt.pkl"))
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

        self.b1 = QtWidgets.QPushButton("Layer Size")
        self.b1.installEventFilter(self)
        self.addWidget(self.b1, 1, 0)

        self.b2 = QtWidgets.QPushButton("Layer Number")
        self.b2.installEventFilter(self)
        self.addWidget(self.b2, 1, 1)

        self.b3 = QtWidgets.QPushButton("Parameter Search Evol")
        self.b3.installEventFilter(self)
        self.addWidget(self.b3, 1, 2)

    def addLegend(self):

        self.l1 = QtWidgets.QLabel()
        self.l1.setText("Metaparameters Visualization")
        self.l1.setAlignment(QtCore.Qt.AlignCenter)
        self.addWidget(self.l1, 0, 0, 1, 3)

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

            if action == "Layer Size":
                self.win.displayStatus("Show the value of the Loss during Test in function of NN Layer Size.")
            if action == "Layer Size":
                self.win.displayStatus("Show the value of the Loss during Test in function of NN number of Layers.")
            if action == "Parameter Search Evol":
                self.win.displayStatus("Show the evolution of the best learning loss along the Hyper-parameters search epochs.")
            return True


class IteButWin(QtWidgets.QGridLayout):

    def __init__(self, win):

        QtWidgets.QGridLayout.__init__(self)
        self.setGeometry(QtCore.QRect(0, 0, 1000, 100))

        self.win = win

        self.addLegend()
        self.addButtons()

    def addButtons(self):

        self.b1 = QtWidgets.QPushButton("Loss")
        self.b1.installEventFilter(self)
        self.addWidget(self.b1, 1, 0)

        self.b2 = QtWidgets.QPushButton("Accuracy")
        self.b2.installEventFilter(self)
        self.addWidget(self.b2, 1, 1)

        self.b3 = QtWidgets.QPushButton("Prediction")
        self.b3.installEventFilter(self)
        self.addWidget(self.b3, 1, 2)

    def addLegend(self):

        self.l1 = QtWidgets.QLabel()
        self.l1.setText("Learning Evolution")
        self.l1.setAlignment(QtCore.Qt.AlignCenter)
        self.addWidget(self.l1, 0, 0, 1, 3)

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

        if "Loss" in action:
            self.win.displayStatus("Plot the Loss evolution during Training and Validation")
        if "Prediction" in action:
            self.win.displayStatus("Plot the predicted signal along with the real test signal")
        if "Accuracy" in action:
            self.win.displayStatus("Plot the Loss evolution during Training and Validation")
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
        self.setWindowTitle("HyQ Learning Viewer")
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
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+C"), self, self.quitUI)
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
