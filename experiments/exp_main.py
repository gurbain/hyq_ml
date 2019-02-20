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
        for t in range(1, 17, 2):
            for l in range(0, 80, 10):
                for kp in np.logspace(1, 3, 16):
                    for kd in np.logspace(-1, 1.4, 8):

                            # Open the config file and retrieve the data
                            config = ConfigParser.ConfigParser()
                            config.read(DEF_CONFIG)
                            config.set("Force", "delay_line_n", str(t))
                            config.set("Force", "delay_line_step", "3")
                            config.set("Force", "elm", "True")
                            config.set("Force", "elm_n", str(l))
                            config.set("Force", "elm_fct", "tanh")
                            config.set("Physics", "noise", "0.0")
                            config.set("Force", "regularization", str(r))
                            config.set("Physics", "init_impedance", str([kp, kd, kp, kd, kp, kd]))

                            # Add it to the experiment list
                            exp_list.append(config)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)
