import ConfigParser
import numpy as np

from hyq import cluster


DEF_CONFIG = "/home/gurbain/hyq_ml/config/sim_config_force_default.txt"


if __name__ == '__main__':

    # Create a docker task manager
    task_manager = cluster.Sequential(n_proc=10)

    # Create a list of experiments
    exp_list = []
    for i in range(10):
        for t in range(1, 41, 2):
            for n in range(0, 400, 40):

                # Open the config file and retrieve the data
                config1 = ConfigParser.ConfigParser()
                config1.read(DEF_CONFIG)
                config1.set("Force", "delay_line_n", str(t))
                config1.set("Physics", "noise", str(n))
                config1.set("Simulation", "inputs", "['bias', 'grf', 'joints']")

                # Without proprioceptive
                config2 = ConfigParser.ConfigParser()
                config2.read(DEF_CONFIG)
                config2.set("Force", "delay_line_n", str(t))
                config2.set("Physics", "noise", str(n))
                config2.set("Simulation", "inputs", "['bias', 'grf']")

                # Add it to the experiment list
                exp_list.append(config1)
                exp_list.append(config2)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)
