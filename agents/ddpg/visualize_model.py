import numpy as np
import gym
import gym_tigrillo
from model import ModelZoo
from ann_visualizer.visualize import ann_viz;




env = gym.make('tigrillo-v0')
np.random.seed(123)
env.seed(123)

#load model
mz = ModelZoo()
nb_actions = env.action_space.shape[0]

mz = ModelZoo()
nb_actions = env.action_space.shape[0]
actor, critic, action_input = mz.create_model(env,'basic')


ann_viz(critic, title="My first neural network")
