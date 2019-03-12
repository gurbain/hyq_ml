import numpy as np

import gym
from gym import wrappers

import time
from multiprocessing import Process, Queue
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import FileLogger,ModelIntervalCheckpoint

from model import ModelZoo
import rospy
import gym_tigrillo
from gym_tigrillo.callbacks import DataLogger
import os
import shutil
from config import ConfigExperiment
import pickle

ENV_IDS = ['20180405-203220','20180405-203211','20180405-203205','20180406-154741','20180406-154848']
ENV_NAMES = ['tigrillo-v20','tigrillo-v22','tigrillo-v25','tigrillo-v26','tigrillo-v27']


experiment_name = 'action_space_4'

test_env_log = []
train_env_log = []
test_history_log = []

def test_agent(q,_env_name,_env_id):
    # load config
    path_config_file = '/experiments/ddpg_{}/app/agents/ddpg/config.json'.format(_env_id)
    ce = ConfigExperiment()
    ce.load_config_json(path_config_file)

    # Get the environment and extract the number of actions.
    ENV_NAME = _env_name
    env = gym.make(ENV_NAME)
    env = wrappers.Monitor(env, 'tmp/{}'.format(ENV_NAME), force=True)
    np.random.seed(123)
    env.seed(123)

    # load model
    mz = ModelZoo()
    nb_actions = env.action_space.shape[0]

    mz = ModelZoo()
    nb_actions = env.action_space.shape[0]
    MODEL_TYPE = ce.get_var('MODEL_TYPE')
    actor, critic, action_input = mz.create_model(env, MODEL_TYPE)

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=ce.get_var('MEMORY_LIMIT'), window_length=ce.get_var('MEMORY_WINDOW'))

    if (ce.get_var('RANDOM_PROCCES') == 'OrnsteinUhlenbeckProcess'):
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
        weights_filename = 'ddpg_{}_weights.h5f'.format(ce.get_var('ENV_NAME'))
        agent.load_weights(weights_filename)
        print("loaded:{}".format(weights_filename))
    except Exception as e:
        print(e)

    try:
        checkpoint_filename = os.path.join(filepath_experiment,
                                           'ddpg_{}_weights_checkpoint.h5f'.format(ce.get_var('ENV_NAME')))
        agent.load_weights(checkpoint_filename)
        print("loaded:{}".format(checkpoint_filename))
    except Exception as e:
        print(e)

    # Finally, evaluate our algorithm for 5 episodes.
    history = agent.test(env, nb_episodes=20, visualize=True, nb_max_episode_steps=2000)
    q.put((history.history, ce.get_var('ENV_NAME')))

for _env_name in ENV_NAMES:
    for _env_id in ENV_IDS:
        queue = Queue()
        p = Process(target=test_agent, args=(queue, _env_name, _env_id))
        p.start()
        p.join()  # this blocks until the process terminates
        history, train_env = queue.get()

        test_env_log.append(_env_name)
        train_env_log.append(train_env)
        test_history_log.append(history)



results = {'test_env_log':test_env_log,
           'train_env_log':train_env_log,
           'test_history_log':test_history_log}
print(results)
path = '/experiments/tests/{}.p'.format(experiment_name)
pickle.dump(results,open(path, "wb"))


