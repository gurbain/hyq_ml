import ConfigParser
import numpy as np

from hyq import cluster


DEF_CONFIG = "/home/gurbain/hyq_ml/config/sim_config_force_default.txt"


if __name__ == '__main__':

    # Create a docker task manager
    task_manager = cluster.Sequential(n_proc=1)

    # Create a list of experiments
    exp_list = []
    for i in range(15):
        for n in range(0, 400, 40):

            # Without high-level inputs
            config1 = ConfigParser.ConfigParser()
            config1.read(DEF_CONFIG)
            config1.set("Physics", "noise", str(n))
            config1.set("Simulation", "inputs", "['bias', 'grf', 'joints']")

            # With high-level inputs
            config2 = ConfigParser.ConfigParser()
            config2.read(DEF_CONFIG)
            config2.set("Physics", "noise", str(n))
            config2.set("Simulation", "inputs", "['bias', 'grf', 'joints', 'imu']")

            # Without high-level inputs and joints
            config3 = ConfigParser.ConfigParser()
            config3.read(DEF_CONFIG)
            config3.set("Physics", "noise", str(n))
            config3.set("Simulation", "inputs", "['bias', 'grf']")

            # With high-level inputs and without joints
            config4 = ConfigParser.ConfigParser()
            config4.read(DEF_CONFIG)
            config4.set("Physics", "noise", str(n))
            config4.set("Simulation", "inputs", "['bias', 'grf', 'imu']")

            # Add it to the experiment list
            exp_list.append(config1)
            exp_list.append(config2)
            exp_list.append(config3)
            exp_list.append(config4)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)