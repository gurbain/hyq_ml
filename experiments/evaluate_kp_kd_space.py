import os
import rospy as ros
import sys
import time

from hyq import utils
from hyq import physics


FOLDER = "/home/gurbain/hyq_ml/data/stifness_space_exploration/" + \
         utils.timestamp() + "/"


def simulate(n, kp, kd):

    # Create and start simulation
    if not ros.is_shutdown():
        p = physics.HyQSim(verbose=1, init_impedance=[kp, kd, kp, kd, kp, kd])
        p.start()

    # Run Simulation for 40 seconds
    t = 0  # sim time
    i = 0  # real timeout
    x = []  # robot init x position
    y = []  # robot init y position
    t_trot = 0 # start of the trotting gait
    trot_flag = False
    while not ros.is_shutdown() and t < 50 and i < 2000:

        # Get simulation time
        t = p.get_sim_time()

        # Start the trot
        if trot_flag is False:
            if p.sim_started:
                trot_flag = p.start_rcf_trot()
                if trot_flag:
                    print " ===== Trotting Started =====\n"
                    t_trot = t

        # Retrieve robot x and y position
        curr_x, curr_y = p.get_hyq_xy()
        x.append(curr_x)
        y.append(curr_y)

        # Count for timeout
        i += 1
        time.sleep(0.1)

    # Get result and stop simulation
    if 'p' in locals():
        p.stop()

        # Save and quit
        to_save = {"t_sim": t, "x": x, "y": y, "t_real": i,
                   "t_trot": t_trot, "index": n, "Kp": kp, "Kd": kd}
        utils.save_on_top(to_save, FOLDER + "results.pkl")

    return


if __name__ == '__main__':

    # Create the folder
    utils.mkdir(FOLDER)
    ros.init_node("physics", anonymous=True)

    # Run space exploration
    n = 1
    # ros.on_shutdown(utils.cleanup)

    for kp in range(50, 1050, 50):
        for kd in range(1, 13, 4):
            for i in range(5):
                if not ros.is_shutdown():
                    print "\n ===== Simulation N=" + str(n) + \
                          " - Kp=" + str(kp) + " - Kd=" + str(kd) + " =====\n"

                    simulate(n, kp, kd)
                    n += 1
