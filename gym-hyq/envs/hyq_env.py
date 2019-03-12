import gym
from gym import spaces
from gym import utils
from gym_hyq import Simulation
import rospy
import numpy as np
from std_msgs.msg import  Float64
from std_srvs.srv import Trigger, Empty
from gazebo_msgs.srv import GetModelState
import time

class TigrilloEnv(gym.Env, utils.EzPickle):

    def __init__(self,model_path_sdf= "/app/data/models/tigrillo_v2/model_feet_v0.2.sdf",
                 control_rate = 25,
                 sim_speed = 1000,
                 sim_speed_adaptive = True,
                 sim_verbose = False):
        #params
        self.control_rate = control_rate
        self.sim_speed = sim_speed
        self.sim_speed_adaptive = sim_speed_adaptive
        self.model_path_sdf = model_path_sdf

        # adaptive simulation speed
        self.time_step_hist = []
        self.time_remaining_hist = []

        # start simulation node
        self.sim = Simulation(verbose=sim_verbose)
        self.sim.start_gazebo()
        self.sim.init_env(self.model_path_sdf)

        # set simulation time
        rospy.set_param('use_sim_time', True)

        # Subscribe to services
        self.init_env_proxy = rospy.ServiceProxy("/simulation/init_env", Trigger)
        self.reset_env_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.get_robot_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    def _seed(self ,seed):

        return 0

    def _close(self):

        print('close')

    def __del__(self):
        """
        Shut down enviroment
        """

        print('shutdown')
        self.sim.stop()
        self.sim.stop_ros()
        del self.sim

    def _step(self, action):
        """
        interact for one step with the enviroment
        """

        # get stats for adaptive simulation speed
        start_time_step = rospy.get_time()
        self.time_step_hist.append(start_time_step)
        self.time_remaining_hist.append(self.rate.remaining().to_sec())

        # get stats for adaptive simulation speed
        self.rate.sleep()

        # send action/ get state, reward
        self._pub_action(action)
        state, reward, episode_over, info = self._get_state_reward_done_info


        return state, reward, episode_over, info

    def _reset(self):
        """
        Reset the enviroment
        """

        self.reset_env_proxy()
        self.rate = rospy.Rate(self.control_rate)

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
                    print("## Speed up from: {} to {}".format(int(prev_speed),int(self.sim_speed)))


                if( self.time_remaining_mean < 0):
                    prev_speed = self.sim_speed
                    self.sim_speed = 0.90 * self.sim_speed
                    print("## Speed down from: {} to {}".format(int(prev_speed),int(self.sim_speed)))


        self.sim.set_physics_properties(self.sim_speed)
        self.time_step_hist = []
        self.time_remaining_hist = []

        time.sleep(0.1)

        self.robot_state_prev = None
        self.reward_prev = None
        state, reward, episode_over,info = self._get_state_reward_done_info

        return state

    def _render(self, mode='human', close=False):
        """
        Viewer only supports human mode currently
        """
        
        self.sim.set_physics_properties(1000)
        self.sim_speed_adaptive = False