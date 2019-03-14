import numpy as np

import gym
from gym import wrappers

import time

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import FileLogger,ModelIntervalCheckpoint

from model import ModelZoo

import gym_tigrillo
from gym_tigrillo.callbacks import DataLogger
import os
import shutil
from config import ConfigExperiment

#load config
ce = ConfigExperiment()
ce.load_config_json('config.json')

# Get the environment and extract the number of actions.
ENV_NAME = ce.get_var('ENV_NAME')
env = gym.make(ENV_NAME)
env = wrappers.Monitor(env, 'tmp/{}'.format(ENV_NAME), force=True)
np.random.seed(123)
env.seed(123)

#load model
mz = ModelZoo()
nb_actions = env.action_space.shape[0]

mz = ModelZoo()
nb_actions = env.action_space.shape[0]
MODEL_TYPE = ce.get_var('MODEL_TYPE')
actor, critic, action_input = mz.create_model(env,MODEL_TYPE)



# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=ce.get_var('MEMORY_LIMIT'), window_length=ce.get_var('MEMORY_WINDOW'))


if(ce.get_var('RANDOM_PROCCES') == 'OrnsteinUhlenbeckProcess'):
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions,
                                              theta=ce.get_var('theta'),
                                              mu=ce.get_var('mu'),
                                              sigma=ce.get_var('sigma'))
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)


agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory,
                  random_process=random_process,
                  nb_steps_warmup_critic=ce.get_var('nb_steps_warmup_critic'),
                  nb_steps_warmup_actor=ce.get_var('nb_steps_warmup_actor'),
                  gamma=ce.get_var('gamma'),
                  target_model_update=ce.get_var('target_model_update'),
                  )

agent.compile([Adam(lr=ce.get_var('lr1')), Adam(lr=ce.get_var('lr1'))], metrics=['mae'])


# load model weights
print('Load model weights')
filepath_experiment = ce.get_var('filepath_experiment')
try:
    weights_filename = os.path.join(filepath_experiment, 'ddpg_{}_weights.h5f'.format(ENV_NAME))
    agent.load_weights(weights_filename)
    print("loaded:{}".format(weights_filename))
except Exception as e:
    print(e)

try:
    checkpoint_filename = os.path.join(filepath_experiment, 'ddpg_{}_weights_checkpoint.h5f'.format(ENV_NAME))
    agent.load_weights(checkpoint_filename)
    print("loaded:{}".format(checkpoint_filename))
except Exception as e:
    print(e)


# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=50, visualize=True, nb_max_episode_steps=2000)