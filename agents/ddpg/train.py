"""
This is the default script to train a NN to control the HyQ robot with DDPG.
It creates an environment (gym_hyq), a neural network (keras) and a reinforcement
learning rule (gym-rl).
Some callbacks are added to save and analyze everything afterwards and the last 
command start the training and save the weights.
"""

import os
import numpy as np

import gym
from gym import wrappers
from gym_hyq.callbacks import DataLogger

from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from model import ModelGenerator
from utils import Config, mkdir, timestamp


EXP_FOLDER = "/home/gurbain/docker_sim/gym_hyq/"


# Get the config and save folder
conf = Config()
exp_filename = EXP_FOLDER + "/ddpg_{}/".format(timestamp())
mkdir(exp_filename)
conf.set('exp_filename', exp_filename)
conf.save_config_json(os.path.join(exp_filename, 'config.json'))


# Set-up Gym Environment
env_name = conf.get('env_name')
env = gym.make(env_name)
env = wrappers.Monitor(env, 'tmp/{}'.format(env_name), force=True)
np.random.seed(123)
env.seed(123)


# Set-up the Keras Neural Networks
mg = ModelGenerator()
actor, critic, action_input = mg.create(env, conf.get('model_type'))


# Set-up Training Agent
memory = SequentialMemory(limit=conf.get('memory_limit'), window_length=conf.get('memory_window'))
random_process = OrnsteinUhlenbeckProcess(size=env.action_space.shape[0],
                                          theta=conf.get('random_theta'), mu=conf.get('random_mu'),  
                                          sigma=conf.get('random_sigma'))
agent = DDPGAgent(nb_actions=env.action_space.shape[0], actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, gamma=conf.get('gamma'), random_process=random_process, 
                  nb_steps_warmup_critic=conf.get('nb_steps_warmup_critic'),
                  nb_steps_warmup_actor=conf.get('nb_steps_warmup_actor'),
                  target_model_update=conf.get('target_model_update'))
agent.compile([Adam(lr=conf.get('lr1')), Adam(lr=conf.get('lr1'))], metrics=[conf.get('metrics')])


# Create training callbacks
log_filename = exp_filename + 'ddpg_{}_log.json'.format(env_name)
chk_filename = os.path.join(exp_filename,'ddpg_{}_weights_checkpoint.h5f'.format(env_name))
wgt_filename = os.path.join(exp_filename,'ddpg_{}_weights.h5f'.format(env_name))

callbacks = [FileLogger(log_filename, interval=1)]
callbacks += [DataLogger(exp_filename, interval=100)]
callbacks += [ModelIntervalCheckpoint(chk_filename, interval=10000, verbose=1)]


# TRAIN and save the weights
agent.fit(env, nb_steps=conf.get('nb_steps'), visualize=False,
          callbacks=callbacks, verbose=2)
agent.save_weights(wgt_filename, overwrite=True)

