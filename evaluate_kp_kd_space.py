import os
import rospy as ros
import sys
import time

import utils
import physics


FOLDER = "/home/gabs48/src/quadruped/hyq/hyq_ml/data/stifness_space_exploration/" + \
         utils.timestamp() + "/"


def simulate(n, kp, kd):

    # Create and start simulation
    if not ros.is_shutdown():
        p = physics.HyQSim(verbose=1, init_impedance=[kp, kd, kp, kd, kp, kd])
        p.start()
        p.register_node()

    # Run Simulation for 40 seconds
    t = 0  # sim time
    i = 0  # real timeout
    t_trot = 0 # start of the trotting gait
    while not ros.is_shutdown() and t < 40 and i < 200:

        # Get simulation time
        t = p.get_sim_time()

        # Start the trot
        if p.sim_started:
            p.start_rcf_trot()
            t_trot = t

        # Count for timeout
        i += 1
        time.sleep(1)

    # Get result and stop simulation
    if 'p' in locals():
        x, y = p.get_hyq_xy()
        p.stop()

        # Save and quit
        to_save = {"t_sim": t, "x": x, "y": y, "t_real": i,
                   "t_trot": t_trot, "index": n, "Kp": kp, "Kd": kd}
        utils.save_on_top(to_save, FOLDER + "results.pkl")

    return


if __name__ == '__main__':

    # Create the folder
    utils.mkdir(FOLDER)

    # Run space exploration
    n = 1
    # ros.on_shutdown(utils.cleanup)

    for kp in range(50, 1050, 50):
        for kd in range(1, 15, 2):
            for i in range(5):
                if not ros.is_shutdown():
                    print " ===== Simulation N=" + str(n) + \
                          " - Kp=" + str(kp) + " - Kd=" + str(kd) + " ====="

                    simulate(n, kp, kd)
                    n += 1
