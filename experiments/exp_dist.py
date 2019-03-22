import ConfigParser
import numpy as np

from cluster import Cluster


DEF_CONFIG = "/home/gurbain/hyq_ml/config/sim_config_force_default.txt"



if __name__ == '__main__':

    # Create a docker task manager
    task_manager = Cluster(n_proc=4)

    # Create a list of experiments
    exp_list = []
    for i in range(2):
        for kp in randomly(np.linspace(75, 4000, 30)):
                    # Open the config file and retrieve the data
                    config1 = ConfigParser.ConfigParser()
                    config1.read(DEF_CONFIG)
                    config1.set("Force", "delay_line_n", str(50))
                    config1.set("Force", "delay_line_step", "1")
                    config1.set("Force", "elm_n", str(50))
                    config1.set("Physics", "init_impedance", str([150, 10, kp, 7.5, kp, 7.5]))

                    # Add it to the experiment list
                    exp_list.append(config1)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)