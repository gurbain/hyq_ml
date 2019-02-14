import ConfigParser
import numpy as np

from hyq import cluster


DEF_CONFIG = "/home/gurbain/hyq_ml/config/sim_config_force_default.txt"


if __name__ == '__main__':

    # Create a docker task manager
    task_manager = cluster.Sequential(n_proc=10)

    # Create a list of experiments
    exp_list = []
    for i in range(2):
        for t in range(1, 20, 1):
            for l in range(0, 80, 5):
                for kp in [10, 50, 100, 250, 500]:
                    for n in [0.0, 50.0, 100.0]:

                        # Open the config file and retrieve the data
                        config = ConfigParser.ConfigParser()
                        config.read(DEF_CONFIG)
                        config.set("Force", "delay_line_n", str(t))
                        config.set("Force", "elm_n", str(l))
                        config.set("Physics", "noise", str(n))
                        config.set("Physics", "init_impedance", str([kp, kp/10, kp, kp/10, kp, kp/10]))

                        # Add it to the experiment list
                        exp_list.append(config)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)
