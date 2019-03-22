"""
Basic Environment for the HyQ Robot. It receives the 12 joints poisition and
the states is a vector made of a bias (dim=1), the real joints values (dim=12),
the ground reaction forces (dim=4), the absolute position and orientation (dim=6).
The reward is set to 1.
"""

import numpy as np

from gym import spaces
from hyq_gym import utils
from hyq_gym.envs import HyQEnv


class HyQBasicEnv(HyQEnv):

    metadata = {'render.modes': ['human']}

    def __init__(self, control_rate=None, sim_speed=None,
                 sim_speed_adaptive=None, sim_verbose=None,
                 sim_view=None, sim_rviz=None, sim_impedance=None,
                 sim_inputs=None):

        # Default ENV parameters
        control_rate = 25 if control_rate is None else control_rate
        sim_speed = 1000 if sim_speed is None else sim_speed
        sim_speed_adaptive = False if sim_speed_adaptive is None else sim_speed_adaptive
        sim_verbose = 1 if sim_verbose is None else sim_verbose
        sim_view = False if sim_view is None else sim_view
        sim_rviz = False if sim_rviz is None else sim_rviz
        sim_impedance = None if sim_impedance is None else sim_impedance
        sim_inputs = ['bias', 'joints', 'grf', 'imu']  if sim_inputs is None else sim_inputs
        
        state_size = 20
        action_size = 12
        
        # Define the ENV space
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(23,), dtype=np.float32)

        HyQEnv.__init__(self, control_rate=control_rate, sim_speed=sim_speed,
                        sim_speed_adaptive=sim_speed_adaptive, sim_verbose=sim_verbose,
                        sim_view=sim_view, sim_rviz=sim_rviz, sim_impedance=sim_impedance,
                        sim_inputs=sim_inputs)

        # Step records
        self.state_prev      = None
        self.reward_prev     = 0
        self.hyq_x_prev      = 0
        self.hyq_y_prev      = 0
        self.hyq_z_prev      = 0
        self.hyq_phi_prev    = 0
        self.hyq_theta_prev  = 0
        self.hyq_psi_prev    = 0
        self.hyq_lf_grf_prev = 0
        self.hyq_rf_grf_prev = 0
        self.hyq_lh_grf_prev = 0
        self.hyq_rh_grf_prev = 0
        self.hyq_fall_prev   = False
        self.hyq_power_prev  = 0

        self.state_scaler = utils.HyQStateScaler()
        self.action_scaler = utils.HyQActionScaler()

    @property
    def _get_state_reward_done_info(self):

        # Get State
        info = {}
        state = self._get_state()

        # Check for crash and re-init if needed

        # Reward
        reward = self._get_reward()

        # Termination
        done = self._get_term()

        # Record step values
        self.state_prev = state
        self.reward_prev = reward
        self.hyq_x_prev = self.sim.hyq_x
        self.hyq_y_prev = self.sim.hyq_y
        self.hyq_z_prev = self.sim.hyq_z
        self.hyq_phi_prev = self.sim.hyq_phi
        self.hyq_theta_prev = self.sim.hyq_theta
        self.hyq_psi_prev = self.sim.hyq_psi
        self.hyq_lf_grf_prev = self.sim.hyq_lf_grf
        self.hyq_rf_grf_prev = self.sim.hyq_rf_grf
        self.hyq_lh_grf_prev = self.sim.hyq_lh_grf
        self.hyq_rh_grf_prev = self.sim.hyq_rh_grf
        self.hyq_fall_prev = self.sim.hyq_fall
        self.hyq_power_prev = self.sim.hyq_power

        return state, reward, done, info


    def _set_action(self, action):

        self.sim.send_hyq_nn_pred(self.action_scaler.inverse_transform(action), 1)

    def _get_reward(self):

        return 1

    def _get_state(self):

        return self.state_scaler.transform(self.sim.get_hyq_state())

    def _get_term(self):

        return self.sim.hyq_fall
