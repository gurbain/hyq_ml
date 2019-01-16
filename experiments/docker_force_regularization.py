
import ConfigParser
import numpy as np

from hyq import cluster


DEF_CONFIG = "/home/gurbain/hyq_ml/config/sim_config_force_default.txt"


if __name__ == '__main__':

    # Create a docker task manager
    task_manager = cluster.Tasks()

    # Create a list of experiments
    exp_list = []
    for i in range(5):
        for r in np.logspace(-6, 1, 8):
            for t in range(1, 16, 3):

                # Open the config file and retrieve the data
                config = ConfigParser.ConfigParser()
                config.read(DEF_CONFIG)
                config.set("Force", "regularization", str(r))
                config.set("Force", "delay_line_n", str(t))
                config.set("Force", "delay_line_step", str(3))

                # Add it to the experiment list
                exp_list.append(config)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)
    print exp_res_dirs
