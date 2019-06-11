
import os
from shutil import copyfile
import sys
import time

from hyq import simulation


DEF_CONFIG = "/home/dls-operator/gabriel_experiment/hyq_ml/configreal_config_force_default.txt"
SAVE_FOLDER = "/home/dls-operator/gabriel_experiment/experiments/"


class Logger(object):

    def __init__(self, file):
        self.terminal = sys.stdout
        self.log = open(file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass    


def timestamp():

    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def mkdir(path):

    if not os.path.exists(path):
        os.makedirs(path)


def create_folder():

    # Create experiment dir
    exp_root_folder = SAVE_FOLDER + "real/"
    mkdir(exp_root_folder)
    exp_dir = exp_root_folder + timestamp()
    mkdir(exp_dir)

    # Add a config file
    copyfile(DEF_CONFIG, exp_dir + "/config.txt")

    return exp_dir


if __name__ == '__main__':

    # Create the experiment folder
    exp_dir = create_folder()

    # Redirect stdout and stderr to log
    sys.stdout = Logger(exp_dir + "/log.txt")
    sys.stderr = Logger(exp_dir + "/log.txt")

    # Run the simulation in that folder
    s = simulation.Simulation(folder=exp_dir)
    s.run()
