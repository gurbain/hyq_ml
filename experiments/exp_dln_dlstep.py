import ConfigParser
import numpy as np

from hyq import cluster


DEF_CONFIG = "/home/gurbain/hyq_ml/config/sim_config_force_default.txt"


if __name__ == '__main__':

    # Create a docker task manager
    task_manager = cluster.Sequential(n_proc=10)

    # Create a list of experiments
    exp_list = []
    for i in range(5):
        for t in range(1, 18, 3):
            for s in range(1, t, 2):

                # Open the config file and retrieve the data
                config = ConfigParser.ConfigParser()
                config.read(DEF_CONFIG)
                config.set("Force", "delay_line_n", str(t))
                config.set("Force", "delay_line_step", str(s))

                # Add it to the experiment list
                exp_list.append(config)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)
