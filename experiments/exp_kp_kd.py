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
        for kp in np.logspace(1, 3, 20):
            for kd in np.logspace(-1, 1.4, 5):

                # Open the config file and retrieve the data
                config1 = ConfigParser.ConfigParser()
                config1.read(DEF_CONFIG)
                config1.set("Physics", "init_impedance", str([kp, kd, kp, kd, kp, kd]))
                config1.set("Simulation", "inputs", "['bias', 'grf', 'joints']")

                # Without proprioceptive
                config2 = ConfigParser.ConfigParser()
                config2.read(DEF_CONFIG)
                config2.set("Physics", "init_impedance", str([kp, kd, kp, kd, kp, kd]))
                config2.set("Simulation", "inputs", "['bias', 'grf']")

                # Add it to the experiment list
                exp_list.append(config1)
                exp_list.append(config2)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)
