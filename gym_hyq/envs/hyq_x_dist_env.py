"""
Sabilization environment for the HyQ Robot.
Same action-state space as the basic ennvironment/
The reward penalizes roll, pitch, yaw, x and y speed and 
position gap around (0, 0, 0.6, 0, 0, 0)
"""

import numpy as np

from gym import spaces
from gym_hyq.envs import HyQBasicEnv


class HyQXDistEnv(HyQBasicEnv):

    def __init__(self):

        # Custom ENV parameters
        control_rate = 25
        sim_speed = 1000
        sim_speed_adaptive = False
        sim_verbose = 1
        sim_view = False
        sim_rviz = False
        sim_impedance = None
        sim_inputs = ['bias', 'imu', 'grf', 'joints']

        HyQBasicEnv.__init__(self, control_rate=control_rate, sim_speed=sim_speed,
                             sim_speed_adaptive=sim_speed_adaptive, sim_verbose=sim_verbose,
                             sim_view=sim_view, sim_rviz=sim_rviz, sim_impedance=sim_impedance,
                             sim_inputs=sim_inputs)

    def _get_reward(self):

    	reward = self.sim.hyq_x - self.hyq_x_prev

        return reward