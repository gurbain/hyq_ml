import gym
from gym import spaces
from gym import utils
from gym_hyq import Simulation
from gym_hyq.envs.hyq_env import TigrilloEnv
import rospy
import time
import math
from sensor_msgs.msg import JointState
from std_msgs.msg import  Float64
from std_srvs.srv import Trigger, Empty
from gazebo_msgs.srv import GetModelState
import tf
import numpy as np


class TigrilloBasicEnv(TigrilloEnv):

    metadata = {'render.modes': ['human']}

    def __init__(self):

        print('init')
        #params
        self.joints = ['right_shoulder',
                       'left_shoulder',
                       'right_hip',
                       'left_hip']
        control_rate = 25
        sim_speed = 1000
        sim_speed_adaptive = True
        # Action/ Observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,))
        self.observation_space = spaces.Box(low=-1, high=1,
                                             shape=(8,))

        # choose model
        self.model_path_sdf ="/app/data/models/tigrillo_call_feet_force/model.sdf"

        #init TrigrilloEnv
        TigrilloEnv.__init__(self,
                             model_path_sdf = self.model_path_sdf,
                             control_rate = control_rate,
                             sim_speed = sim_speed,
                             sim_speed_adaptive = sim_speed_adaptive)


        # Subscribe to ros topics
        self.joint_states_sub = rospy.Subscriber('/joint_states',JointState,self._get_jointState_callback)
        self._jointStateMessage = None

        # Publish ros topic
        self.pub_leg_fl = rospy.Publisher('/tigrillo/left_shoulder/cmd_pos', Float64, queue_size=1)
        self.pub_leg_fr = rospy.Publisher('/tigrillo/right_shoulder/cmd_pos', Float64, queue_size=1)
        self.pub_leg_bl = rospy.Publisher('/tigrillo/left_hip/cmd_pos', Float64, queue_size=1)
        self.pub_leg_br = rospy.Publisher('/tigrillo/right_hip/cmd_pos', Float64, queue_size=1)

    def _pub_action(self, action):
        # self.action_publisher.publish(Float64MultiArray(data=self.action))
        self.pub_leg_fl.publish(Float64(action[0]))
        self.pub_leg_fr.publish(Float64(action[1]))
        self.pub_leg_bl.publish(Float64(action[2]))
        self.pub_leg_br.publish(Float64(action[3]))


    @property
    def _get_state_reward_done_info(self):

        ##############################
        ##### get robot state
        ##############################
        info = {}
        state = []
        state_dim = 0

        robot_state = self.get_robot_state_proxy('tigrillo', '')


        # get joint properties
        joint_positions = []
        joint_rates = []
        joint_efforts = []
        for name,pos,vel,effort in zip(self._jointStateMessage.name, self._jointStateMessage.position, self._jointStateMessage.velocity, self._jointStateMessage.effort):
            if(name in self.joints):
                joint_positions.append(pos)
                joint_rates.append(vel)
                joint_efforts.append(effort)

        state = np.hstack((state,joint_positions))
        state_dim += len(joint_positions)
        state = np.hstack((state,np.multiply(joint_rates, 0.01)))
        state_dim += len(joint_rates)



        ###############################
        ### check for crash model
        ###############################
        # @TODO make this better practice and log if crash
        if (math.isnan(joint_rates[0])):
            self.sim.init_env(self.model_path_sdf)
            info['env_reset'] = 'joint_rates = nan'
            time.sleep(10)


        if (not self.robot_state_prev):
            self.robot_state_prev = robot_state
            self.joint_positions_prev = joint_positions
            self.joint_rates_prev = joint_rates

        # Robot position
        robot_position = robot_state.pose.position
        robot_position_prev = self.robot_state_prev.pose.position

        #@TODO check best reward
        d_distance = math.sqrt(
            math.pow(robot_position.x - robot_position_prev.x, 2) + math.pow(robot_position.y - robot_position_prev.y,
                                                                             2))
        # d_distance = (robot_position.x - robot_position_prev.x)


        # Robot rotation
        robot_orientation = robot_state.pose.orientation
        quaternion = (
            robot_orientation.x,
            robot_orientation.y,
            robot_orientation.z,
            robot_orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]


        ##############################
        ##### get robot reward
        ##############################

        #calc power
        power_array = np.multiply(joint_rates,joint_efforts)
        power_total = np.sum(abs(power_array))

        info['power'] = power_total
        info['d_distance'] = d_distance

        #reward
        Sref = 0.2
        reward = d_distance / Sref

        ##############################
        ##### get robot done
        ##############################
        done = False
        if (abs(roll) > math.pi / 2):
            done = True
            reward = -1.
        elif (abs(pitch) > math.pi / 2):
            done = True
            reward = -1.


        self.robot_state_prev = robot_state
        self.joint_positions_prev = joint_positions
        self.joint_rates_prev = joint_rates
        self.reward_prev = reward

        return state, reward, done, info

    def _get_jointState_callback(self, message):

        self._jointStateMessage = message
