import numpy as np
import os
import rospy as ros
import sys
import time

from hyq import utils
from hyq import physics


FOLDER = "/home/gurbain/hyq_ml/data/stiffness_space_exploration/" + \
         utils.timestamp() + "/"


def simulate(n, kp, kd):

    # Create and start simulation
    if not ros.is_shutdown():
        p = physics.HyQSim(verbose=1, init_impedance=[kp, kd, kp, kd, kp, kd])
        p.start()

    # Run Simulation for 40 seconds
    t = 0  # sim time
    i = 0  # real timeout
    x = []      # robot init x position
    y = []      # robot init y position
    z = []      # robot init z position
    phi = []    # robot init roll
    theta = []  # robot init pitch
    psi = []    # robot init yaw
    t_trot = 0  # start of the trotting gait
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
        curr_x, curr_y, curr_z = p.get_hyq_x_y_z()
        curr_phi, curr_theta, curr_psi = p.get_hyq_phi_theta_psi()
        x.append(curr_x)
        y.append(curr_y)
        z.append(curr_z)
        phi.append(curr_phi)
        theta.append(curr_theta)
        psi.append(curr_psi)

        # Count for timeout
        i += 1
        time.sleep(0.1)

    # Get result and stop simulation
    if 'p' in locals():
        p.stop()

        # Save and quit
        to_save = {"t_sim": t, "t_real": i, "t_trot": t_trot, 
                   "index": n, "Kp": kp, "Kd": kd,
                   "x": x, "y": y, "z": z, 
                   "phi": phi, "theta": theta, "psi": psi}
        utils.save_on_top(to_save, FOLDER + "results.pkl")

    return


if __name__ == '__main__':

    # Create the folder
    utils.mkdir(FOLDER)
    ros.init_node("physics", anonymous=True)

    # Run space exploration
    n = 1
    # ros.on_shutdown(utils.cleanup)

    for i in range(4):
        for kp in np.logspace(0, 3, 30):
            for kd in np.logspace(0, 2, 10):
                if not ros.is_shutdown():
                    print "\n ===== Simulation N=" + str(n) + \
                          " - Kp=" + str(kp) + " - Kd=" + str(kd) + " =====\n"
                    simulate(n, kp, kd)
                    n += 1
