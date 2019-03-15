"""
Basic Environment for the HyQ Robot. It receives the 12 joints poisition and
the states is a vector made of a bias (dim=1), the real joints values (dim=12),
the ground reaction forces (dim=4), the absolute position and orientation (dim=6).
The reward is set to 1.
"""

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from gym import spaces
from gym_hyq.envs import HyQEnv


class HyQStateScaler(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.n_in = 0
        self.mins = np.array([-1, -100000, -100000, -100000, -100000,
                              -0.3, 0, -2, -0.3, 0, -2, -0.3, -1, 0, -0.3, -1, 0,
                              -1.6, -3, -1.6, -1000, -1000, 0])
        self.maxs = np.array([1, 100000, 100000, 100000, 100000,
                              0.1, 1, 0, 0.1, 1, 0, 0.1, 0, 2, 0.1, 0, 2,
                              1.6, 3, 1.6, 1000, 1000, 1])
        self.x_scaled = None

    def _fit_transform(self, x):

        if isinstance(x, list):
            x = np.array(x)
        self.n_in = x.shape[0]

        x_std = (x - self.mins) / (self.maxs - self.mins)
        self.x_scaled = x_std * 2 - np.ones(x_std.shape)

        return self

    def fit(self, x):

        self = self._fit_transform(x)
        return self

    def fit_transform(self, x, y=None, **kwargs):

        self = self._fit_transform(x)
        return self.x_scaled

    def transform(self, x):

        self = self._fit_transform(x)
        return self.x_scaled

    def inverse_transform(self, x):

        if isinstance(x, list):
            x = np.array(x)

        x_std = (x + np.ones(x.shape)) / 2
        return x_std * (self.maxs - self.mins) + self.mins


class HyQActionScaler(HyQStateScaler):

    def __init__(self):

        HyQStateScaler.__init__(self)

        self.mins = np.array([-0.3, 0, -2, -0.3, 0, -2, -0.3, -1, 0, -0.3, -1, 0])
        self.maxs = np.array([0.1, 1, 0, 0.1, 1, 0, 0.1, 0, 2, 0.1, 0, 2])


class HyQBasicEnv(HyQEnv):

    metadata = {'render.modes': ['human']}

    def __init__(self, control_rate=None, sim_speed=None,
                 sim_speed_adaptive=None, sim_verbose=None,
                 sim_view=None, sim_rviz=None, sim_impedance=None,
                 sim_inputs=None):

        # Default ENV parameters
        control_rate = 25 if control_rate is None else control_rate
        sim_speed = 1000 if sim_speed is None else sim_speed
        sim_speed_adaptive = True if sim_speed_adaptive is None else sim_speed_adaptive
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

        self.state_scaler = HyQStateScaler()
        self.action_scaler = HyQActionScaler()

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




if __name__ == "__main__":

    ss = HyQStateScaler()
    sa = HyQActionScaler()

    print sa.inverse_transform([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    print sa.inverse_transform([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    print sa.inverse_transform([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    print sa.transform([-0.2, 0.6, -1.7, -0.2, 0.6, -1.7, -0.2, -0.6, 1.7,-0.2, -0.6, 1.7])

    print ss.transform([0, -100000, -100000, -100000, -100000,
                        -0.3, 0, -2, -0.3, 0, -2, -0.3, -1, 0, -0.3, -1, 0,
                        -1.6, -3, -1.6, -1000, -1000, 0])
    print ss.transform([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    print ss.transform([1, 100000, 100000, 100000, 100000,
                        0.1, 1, 0, 0.1, 1, 0, 0.1, 0, 2, 0.1, 0, 2,
                        1.6, 3, 1.6, 1000, 1000, 1])