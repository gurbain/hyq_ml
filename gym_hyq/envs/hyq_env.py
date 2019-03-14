import gym
from gym import spaces
from gym import utils
from gym_hyq import HyQSim
import rospy as ros
import numpy as np
from std_msgs.msg import  Float64
from std_srvs.srv import Trigger, Empty
from gazebo_msgs.srv import GetModelState
import time

class HyQEnv(gym.Env, utils.EzPickle):

    def __init__(self, control_rate=25, sim_speed=1000, sim_speed_adaptive=True,
                 sim_verbose=0, sim_view=False, sim_rviz=False,
                 sim_impedance=None, sim_inputs=None):

        #params
        self.control_rate = control_rate
        self.sim_speed = sim_speed
        self.sim_speed_adaptive = sim_speed_adaptive
        self.sim_verbose = sim_verbose
        self.sim_view = sim_view
        self.sim_rviz = sim_rviz
        self.sim_impedance = sim_impedance
        self.sim_inputs = sim_inputs

        # adaptive simulation speed
        self.time_step_hist = []
        self.time_remaining_hist = []

        # start simulation node
        self.sim = HyQSim(view=self.sim_view, rviz=self.sim_rviz,
                          init_impedance=self.sim_impedance, verbose=self.sim_verbose,
                          inputs=self.sim_inputs)

        self.sim.start_ros(self.sim_verbose==2)
        ros.init_node("simulation", anonymous=True)
        self.sim.start()
        self._init_rcf()

        # set simulation time
        ros.set_param('use_sim_time', True)

        # Subscribe to services
        self.init_env_proxy = ros.ServiceProxy("/simulation/init_env", Trigger)
        self.reset_env_proxy = ros.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.get_robot_state_proxy = ros.ServiceProxy('/gazebo/get_model_state', GetModelState)

    def _init_rcf(self):

        while not ros.is_shutdown():
            self.t = self.sim.get_sim_time()
            if self.t > 1 and self.sim.controller_started:
                break

    def _seed(self ,seed):

        return 0

    def _close(self):

        print('close')

    def __del__(self):
        """
        Shut down enviroment
        """

        self.sim.stop()
        self.sim.stop_ros()
        self.printv("[HyQ Gym] Gym is stopped")
        del self.sim

    def step(self, action):
        """
        interact for one step with the enviroment
        """

        # get stats for adaptive simulation speed
        start_time_step = ros.get_time()
        self.time_step_hist.append(start_time_step)
        self.time_remaining_hist.append(self.rate.remaining().to_sec())

        # get stats for adaptive simulation speed
        self.rate.sleep()

        # send action/ get state, reward
        self._set_action(action)
        state, reward, episode_over, info = self._get_state_reward_done_info

        return state, reward, episode_over, info

    def reset(self):
        """
        Reset the enviroment
        """

        self.reset_env_proxy()
        self.rate = ros.Rate(self.control_rate)

        # adaptive speedup
        if(self.time_step_hist):
            step_time = 1. / self.control_rate
            step_times = np.diff(self.time_step_hist)
            rate = 1. / np.array(np.diff(step_times))

            self.step_rate_mean = np.mean(rate)
            self.step_rate_std = np.std(rate)
            self.time_remaining_mean = np.mean(self.time_remaining_hist)
            self.step_time_rmse =  np.mean(np.sqrt(np.power(step_times - step_time,2)))
            # print('Stats episode: mean rate:{} std rate:{} mean time remaining:{}'.format(self.step_rate_mean,self.step_rate_std,self.time_remaining_mean))
            if( self.sim_speed_adaptive):

                if( self.time_remaining_mean > (0.3 * step_time)):
                    prev_speed = self.sim_speed
                    # factor = max(min(1, abs(self.time_remaining_mean / step_time)), 0)
                    self.sim_speed = self.sim_speed * 1.05
                    print("[HyQ Gym] Speed up physics update rate from: {} to {}".format(int(prev_speed),int(self.sim_speed)))


                if( self.time_remaining_mean < 0):
                    prev_speed = self.sim_speed
                    self.sim_speed = 0.90 * self.sim_speed
                    print("[HyQ Gym] Speed down physics update rate from: {} to {}".format(int(prev_speed),int(self.sim_speed)))


        self.sim.set_phys_prop(self.sim_speed)
        self.time_step_hist = []
        self.time_remaining_hist = []

        time.sleep(0.1)

        self.robot_state_prev = None
        self.reward_prev = None
        state, reward, episode_over,info = self._get_state_reward_done_info

        return state

    def seed(self, seed):
        
        return 0

    def _render(self, mode='human', close=False):
        """
        Viewer only supports human mode currently
        """
        
        self.sim.set_phys_prop(1000)
        self.sim_speed_adaptive = False

    def printv(self, txt):

        if self.sim_verbose >= 1:
            print(txt)