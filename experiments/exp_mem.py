import ConfigParser
import numpy as np
import random

from hyq import cluster
from hyq.utils import randomly

DEF_CONFIG = "/home/gurbain/hyq_ml/config/sim_config_force_default.txt"


if __name__ == '__main__':

    # Create a docker task manager
    task_manager = cluster.Sequential(n_proc=10)

    # Create a list of experiments
    exp_list = []
    for i in range(10):
        for n in randomly(range(0, 100, 40)):
            for t in radomly(range(1, 80, 1)):

                config1 = ConfigParser.ConfigParser()
                config1.read(DEF_CONFIG)
                config1.set("Force", "delay_line_n", str(t))
                config1.set("Force", "delay_line_step", "1")
                config1.set("Physics", "noise", str(n))
                config1.set("Simulation", "inputs", "['bias', 'grf']")

                config2 = ConfigParser.ConfigParser()
                config2.read(DEF_CONFIG)
                config2.set("Force", "delay_line_n", str(t))
                config2.set("Force", "delay_line_step", "3")
                config2.set("Physics", "noise", str(n))
                config2.set("Simulation", "inputs", "['bias', 'grf']")

                # Add it to the experiment list
                exp_list.append(config1)
                exp_list.append(config2)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)
