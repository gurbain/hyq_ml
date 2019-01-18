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
            for n in np.hstack((np.array([0]), np.logspace(0, 2, 5))):
                for p in range(25, 100, 25):

                    print " ----->> i=" + str(i) + " t=" + str(t) + \
                          " n=" + str(n) + " p=" + str(p)

                    # Open the config file and retrieve the data
                    config = ConfigParser.ConfigParser()
                    config.read(DEF_CONFIG)
                    config.set("Force", "delay_line_n", str(t))
                    config.set("Physics", "noise", str(n))
                    config.set("Physics", "noise_it_min", str(p))
                    config.set("Physics", "noise_it_max", str(p*10))

                    # Add it to the experiment list
                    exp_list.append(config)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)
