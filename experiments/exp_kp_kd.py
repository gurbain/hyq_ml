import ConfigParser
import numpy as np
import random

from hyq import cluster


DEF_CONFIG = "/home/gurbain/hyq_ml/config/sim_config_force_default.txt"


def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)


if __name__ == '__main__':

    # Create a docker task manager
    task_manager = cluster.Sequential(n_proc=10)

    # Create a list of experiments
    exp_list = []
    for i in range(1):
        for kp in randomly(np.logspace(0, 4, 60)):
            for kd in randomly(np.logspace(-1, 3, 60)):

                # Open the config file and retrieve the data
                config1 = ConfigParser.ConfigParser()
                config1.read(DEF_CONFIG)
                config1.set("Physics", "init_impedance", str([150, 10, kp, kd, kp, kd]))

                # Add it to the experiment list
                exp_list.append(config1)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)