import ConfigParser
import numpy as np

from hyq import cluster


DEF_CONFIG = "/home/gurbain/hyq_ml/config/sim_config_force_default.txt"


if __name__ == '__main__':

    # Create a docker task manager
    task_manager = cluster.Sequential(n_proc=10)

    # Create a list of experiments
    exp_list = []
    for i in range(7):
        for t in range(1, 31, 2):
            for l in range(1, 100, 5):
                for kp in [10, 50, 100, 250, 500]:

                    # Open the config file and retrieve the data
                    config1 = ConfigParser.ConfigParser()
                    config1.read(DEF_CONFIG)
                    config1.set("Force", "delay_line_n", str(t))
                    config1.set("Force", "elm_n", str(l))
                    config1.set("Simulation", "inputs", "['bias', 'grf', 'joints']")
                    config1.set("Physics", "init_impedance", str([kp, kp/10, kp, kp/10, kp, kp/10]))

                    config2 = ConfigParser.ConfigParser()
                    config2.read(DEF_CONFIG)
                    config2.set("Force", "delay_line_n", str(t))
                    config2.set("Force", "elm_n", str(l))
                    config2.set("Simulation", "inputs", "['bias', 'grf']")
                    config2.set("Physics", "init_impedance", str([kp, kp/10, kp, kp/10, kp, kp/10]))

                    # Add it to the experiment list
                    exp_list.append(config1)
                    exp_list.append(config2)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)