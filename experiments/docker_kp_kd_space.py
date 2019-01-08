
import ConfigParser
import numpy as np

from hyq import cluster


DEF_CONFIG = "/home/gurbain/hyq_ml/config/sim_config_default.txt"


if __name__ == '__main__':

    # Create a docker task manager
    task_manager = cluster.Tasks()

    # Create a list of experiments
    exp_list = []
    for i in range(4):
        for kp in np.logspace(0, 3, 30):
            for kd in np.logspace(0, 2, 10):

                # Open the config file and retrieve the data
                config = ConfigParser.ConfigParser()
                config.read(DEF_CONFIG)
                config.set("Debug", "verbose", "2")
                config.set("Simulation", "save", "False")
                config.set("Physics", "save", "True")
                config.set("Physics", "init_impedance", str([kp, kd, kp, kd, kp, kd]))

                # Add it to the experiment list
                exp_list.append(config)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)
    print exp_res_dirs
