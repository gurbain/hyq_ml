import ConfigParser
import numpy as np

from cluster import Cluster


DEF_CONFIG = "/home/gabs48/hyq_ml/config/default_conf.txt"


if __name__ == '__main__':

    # Create a docker task manager
    task_manager = Cluster(n_proc=2)

    # Create a list of experiments
    exp_list = []
    for i in range(2):
        for lr1 in np.logspace(-6, -1, 6):
            for lr2 in np.logspace(-6, -1, 6):

                conf = ConfigParser.ConfigParser()
                conf.read(DEF_CONFIG)
                conf.set("Agent", "lr1", str(lr1))
                conf.set("Agent", "lr2", str(lr2))

                # Add it to the experiment list
                exp_list.append(conf)

    # Process all in parallel
    print "[HyQ Exp] Starting Cluster Experiment with " + str(len(exp_list)) + " simulations"
    exp_res_dirs = task_manager.process(exp_list)