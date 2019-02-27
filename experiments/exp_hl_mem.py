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
    for i in range(15):
        for t in randomly(range(1, 100, 2)):

            # Without high-level inputs
            config1 = ConfigParser.ConfigParser()
            config1.read(DEF_CONFIG)
            config1.set("Simulation", "inputs", "['bias', 'grf', 'joints']")
            config1.set("Force", "delay_line_n", str(t))

            # With high-level inputs
            config2 = ConfigParser.ConfigParser()
            config2.read(DEF_CONFIG)
            config2.set("Simulation", "inputs", "['bias', 'grf', 'joints', 'imu']")
            config2.set("Force", "delay_line_n", str(t))

            # Without high-level inputs and joints
            config3 = ConfigParser.ConfigParser()
            config3.read(DEF_CONFIG)
            config3.set("Simulation", "inputs", "['bias', 'grf']")
            config3.set("Force", "delay_line_n", str(t))

            # With high-level inputs and without joints
            config4 = ConfigParser.ConfigParser()
            config4.read(DEF_CONFIG)
            config4.set("Simulation", "inputs", "['bias', 'grf', 'imu']")
            config4.set("Force", "delay_line_n", str(t))

            # Add it to the experiment list
            exp_list.append(config1)
            exp_list.append(config2)
            exp_list.append(config3)
            exp_list.append(config4)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)
