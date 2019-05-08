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
    for i in range(10):
        for kp in randomly(np.linspace(75, 4000, 30)):
                    # Open the config file and retrieve the data
                    config1 = ConfigParser.ConfigParser()
                    config1.read(DEF_CONFIG)
                    config1.set("Force", "delay_line_n", str(50))
                    config1.set("Force", "delay_line_step", "1")
                    config1.set("Force", "elm_n", str(40))
                    config1.set("Physics", "init_impedance", str([150, 10, kp, 7.5, kp, 7.5]))

                    # Add it to the experiment list
                    exp_list.append(config1)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)