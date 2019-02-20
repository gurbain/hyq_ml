import ConfigParser
import numpy as np

from hyq import cluster


DEF_CONFIG = "/home/gurbain/hyq_ml/config/sim_config_force_default.txt"


if __name__ == '__main__':

    # Create a docker task manager
    task_manager = cluster.Sequential(n_proc=10)

    # Create a list of experiments
    exp_list = []
    for i in range(3):
        for l in range(0, 80, 5):
            for kp in np.logspace(0.8, 3.2, 20):
                for kd in np.logspace(-1, 1.4, 4):
                    for n in [0.0, 50.0, 100.0]:
                        # Open the config file and retrieve the data
                        config = ConfigParser.ConfigParser()
                        config.read(DEF_CONFIG)
                        config.set("Force", "elm_n", str(l))
                        config.set("Physics", "noise", str(n))
                        config.set("Physics", "init_impedance", str([kp, kd, kp, kd, kp, kd]))

                        # Add it to the experiment list
                        exp_list.append(config)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)