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
        for kp in randomly([75, 250, 1000, 4000]):
            for l in randomly(np.linspace(1, 120, 30)):
                for t in randomly(np.linspace(1, 100, 30)):

                    # Open the config file and retrieve the data
                    config1 = ConfigParser.ConfigParser()
                    config1.read(DEF_CONFIG)
                    config1.set("Force", "delay_line_n", str(int(t)))
                    config1.set("Force", "delay_line_step", "1")
                    config1.set("Force", "elm_n", str(int(l)))
                    config1.set("Physics", "init_impedance", str([150, 10, kp, 7.5, kp, 7.5]))

                    # Add it to the experiment list
                    exp_list.append(config1)

    print len(exp_list)
    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)