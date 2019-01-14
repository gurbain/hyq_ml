
import ConfigParser
import numpy as np

from hyq import cluster


DEF_CONFIG = "/home/gurbain/hyq_ml/config/sim_config_force_default.txt"


if __name__ == '__main__':

    # Create a docker task manager
    task_manager = cluster.Tasks()

    # Create a list of experiments
    times = [[20, 10], [20, 15], [20, 20], [20, 25], [20, 30],
             [40, 30], [40, 35], [40, 40], [40, 45], [40, 50],
             [80, 70], [80, 75], [80, 80], [80, 85], [80, 90]]
    reg = np.logspace(-6, 1, 8)
    exp_list = []
    for i in range(5):
        for r in reg:
            for t in times:

                # Open the config file and retrieve the data
                config = ConfigParser.ConfigParser()
                config.read(DEF_CONFIG)
                config.set("Force", "regularization", str(r))
                config.set("Force", "delay_line_n", str(15))
                config.set("Force", "delay_line_step", str(3))
                config.set("Timing", "t_start_cl", str(8))
                config.set("Timing", "t_train", str(t[0]))
                config.set("Timing", "t_stop_cl", str(t[1]))
                config.set("Timing", "t_sim", str(min(t[0], t[1]) + 20))

                # Add it to the experiment list
                exp_list.append(config)

    # Process all in parallel
    exp_res_dirs = task_manager.process(exp_list)
    print exp_res_dirs
