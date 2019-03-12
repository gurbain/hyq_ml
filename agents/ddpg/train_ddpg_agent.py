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
from rl.processors import WhiteningNormalizerProcessor

from model import ModelZoo

import gym_tigrillo
from gym_tigrillo.callbacks import DataLogger
import os
import shutil
from config import ConfigExperiment

ce = ConfigExperiment()

# Get the environment and extract the number of actions.
ENV_NAME = ce.get_var('ENV_NAME')
env = gym.make(ENV_NAME)
env = wrappers.Monitor(env, 'tmp/{}'.format(ENV_NAME), force=True)
np.random.seed(123)
env.seed(123)


# save experiment vars

timestr = time.strftime("%Y%m%d-%H%M%S")
filepath_experiment = "/experiments/ddpg_{}/".format(timestr)

#copy trainfile to experiment path

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

srcfile = '/app/'
dstdir = filepath_experiment + 'app/'
copytree(srcfile, dstdir)

#save config file
ce.set_var('filepath_experiment',filepath_experiment)
ce.save_config_json(os.path.join(filepath_experiment,'app','agents','ddpg','config.json'))


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

if(ce.get_var('WhiteningNormalizerProcessor')):
    processor = WhiteningNormalizerProcessor()
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory,
                      random_process=random_process,
                      nb_steps_warmup_critic=ce.get_var('nb_steps_warmup_critic'),
                      nb_steps_warmup_actor=ce.get_var('nb_steps_warmup_actor'),
                      gamma=ce.get_var('gamma'),
                      target_model_update=ce.get_var('target_model_update'),
                      processor=processor
                      )
else:
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory,
                      random_process=random_process,
                      nb_steps_warmup_critic=ce.get_var('nb_steps_warmup_critic'),
                      nb_steps_warmup_actor=ce.get_var('nb_steps_warmup_actor'),
                      gamma=ce.get_var('gamma'),
                      target_model_update=ce.get_var('target_model_update'),
                      )


agent.compile([Adam(lr=ce.get_var('lr1')), Adam(lr=ce.get_var('lr1'))], metrics=['mae'])

##Make callbacks

callbacks = []
log_filename = filepath_experiment + 'ddpg_{}_log.json'.format(ENV_NAME)
callbacks += [FileLogger(log_filename, interval=1)]

# log all train data with custom callback
callbacks += [DataLogger(filepath_experiment,interval=100)]

# make model checkpoints
checkpoint_filename = os.path.join(filepath_experiment,'ddpg_{}_weights_checkpoint.h5f'.format(ENV_NAME))
callbacks += [ModelIntervalCheckpoint(checkpoint_filename,interval=10000,verbose = 1)]


agent.fit(env, nb_steps=10000000, visualize=False, callbacks = callbacks, verbose=2)

# After training is done, we save the final weights.
weights_filename = os.path.join(filepath_experiment,'ddpg_{}_weights.h5f'.format(ENV_NAME))
agent.save_weights(weights_filename, overwrite=True)

