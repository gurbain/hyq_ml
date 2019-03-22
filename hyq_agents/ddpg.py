"""
This is the default script to control the HyQ robot with DDPG.
It creates an environment (hyq_gym), a neural network (keras) and a reinforcement
learning rule (gym-rl).
Some callbacks are added to save and analyze everything afterwards and the last
command start the training and save the weights.
"""


import os

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate

from hyq_gym.callbacks import DataLogger
import utils


class DDPG():

    def __init__(self, env, config):

        self.env = env
        self.config = config

        # Names
        self.env_name = self.config['env_name']
        self.sim_folder = self.config['sim_folder']
        self.log_filename = os.path.join(self.sim_folder,
                                         'ddpg_{}_log.json'.format(self.env_name))
        self.chk_filename = os.path.join(self.sim_folder,
                                         'ddpg_{}_weights_checkpoint.h5f'.format(self.env_name))
        self.wgt_filename = os.path.join(self.sim_folder,
                                         'ddpg_{}_weights.h5f'.format(self.env_name))
        self.log_dir = os.path.join(self.sim_folder,
                                    'tensorboard_{}'.format(self.env_name))

        # Class objects
        self.actor = None
        self.critic = None
        self.action_input = None
        self.memory = None
        self.random_process = None
        self.callbacks = None

        self.__create()

    def fit(self):

        # Create training callbacks
        self.callbacks = [FileLogger(self.log_filename, interval=1)]
        self.callbacks += [DataLogger(self.sim_folder, interval=100)]
        self.callbacks += [ModelIntervalCheckpoint(self.chk_filename, interval=10000, verbose=1)]
        self.callbacks += [TensorBoard(log_dir=self.log_dir, write_graph=True)]

        # Train
        print("\n\n[DDPG] Train agent")
        self.agent.fit(self.env,
                       nb_steps=int(self.config['nb_steps']),
                       visualize=False,
                       callbacks=self.callbacks,
                       verbose=2)

        # Save the weights
        self.agent.save_weights(self.wgt_filename, overwrite=True)

    def test(self):

        print self.wgt_filename.replace('.h5f', '_actor.h5f')
        print utils.is_file(self.wgt_filename.replace('.h5f', '_actor.h5f'))
        if utils.is_file(self.wgt_filename.replace('.h5f', '_actor.h5f')):
            f = self.wgt_filename
        elif utils.is_file(self.chk_filename.replace('.h5f', '_actor.h5f')):
            f = self.chk_filename
        else:
            print("\n\n[DDPG] No weight file. Finishing")
            return

        print("\n\n[DDPG] Load weights from " + f)
        try:
            self.agent.load_weights(f)
        except Exception as e:
            print(e)

        print("\n\n[DDPG] Testing robot")
        self.agent.test(self.env,
                        nb_episodes=50,
                        visualize=True,
                        nb_max_episode_steps=2000)

    def __create(self):

        # Set-up the Keras Neural Networks
        print("\n\n[DDPG] Create Models")
        self.__create_model()

        # Set-up and compile the training Agent
        print("\n\n[DDPG] Create and compile Agent")
        self.memory = SequentialMemory(limit=int(self.config['memory_limit']),
                                       window_length=int(self.config['memory_window']))
        self.random_process = OrnsteinUhlenbeckProcess(size=self.env.action_space.shape[0],
                                                       theta=float(self.config['random_theta']),
                                                       mu=float(self.config['random_mu']),
                                                       sigma=float(self.config['random_sigma']))
        self.agent = DDPGAgent(nb_actions=self.env.action_space.shape[0],
                               actor=self.actor,
                               critic=self.critic,
                               critic_action_input=self.action_input,
                               memory=self.memory,
                               gamma=float(self.config['gamma']),
                               random_process=self.random_process,
                               nb_steps_warmup_critic=int(self.config['nb_steps_warmup_critic']),
                               nb_steps_warmup_actor=int(self.config['nb_steps_warmup_actor']),
                               target_model_update=float(self.config['target_model_update']))
        self.agent.compile([Adam(lr=float(self.config['lr1'])),
                            Adam(lr=float(self.config['lr2']))],
                           metrics=[self.config['metrics']])

    def __create_model(self):

        if self.config['model_type'] == 'basic':
            self.__create_model_basic()

        elif self.config['model_type'] == 'basic_3_layer':
            return self.__create_model_3_layers()

        else:
            raise Exception('ModelError', 'Model \'' + self.config['model_type'] + '\' not found')

    def __create_model_basic(self):

        assert len(self.env.action_space.shape) == 1
        nb_actions = self.env.action_space.shape[0]

        self.actor = Sequential()
        self.actor.add(Flatten(input_shape=(1, 1,) + self.env.observation_space.shape))
        self.actor.add(Dense(64))
        self.actor.add(Activation('relu'))
        self.actor.add(Dense(32))
        self.actor.add(Activation('relu'))
        self.actor.add(Dense(nb_actions))
        self.actor.add(Activation('tanh'))
        print("\n\n[DDPG] Actor Network: ")
        print(self.actor.summary())

        self.action_input = Input(shape=(nb_actions,), name='action_input')
        state_input = Input(shape=(1, 1,) + self.env.observation_space.shape, name='state_input')
        flattened_state = Flatten()(state_input)
        x = Dense(64)(flattened_state)
        x = Activation('relu')(x)
        x = Concatenate()([x, self.action_input])
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        self.critic = Model(inputs=[self.action_input, state_input], outputs=x)
        print("\n\n[DDPG] Critic Network: ")
        print(self.critic.summary())

    def __create_model_3_layers(self):

        assert len(self.env.action_space.shape) == 1
        nb_actions = self.env.action_space.shape[0]

        # Next, we build a very simple model.
        self.actor = Sequential()
        self.actor.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
        self.actor.add(Dense(128))
        self.actor.add(Activation('relu'))
        self.actor.add(Dense(64))
        self.actor.add(Activation('relu'))
        self.actor.add(Dense(32))
        self.actor.add(Activation('relu'))
        self.actor.add(Dense(nb_actions))
        self.actor.add(Activation('tanh'))
        print("\n\n[DDPG] Actor Network: ")
        print(self.actor.summary())

        self.action_input = Input(shape=(nb_actions,), name='action_input')
        state_input = Input(shape=(1,) + self.env.observation_space.shape, name='state_input')
        flattened_state = Flatten()(state_input)
        x = Dense(128)(flattened_state)
        x = Activation('relu')(x)
        x = Concatenate()([x, self.action_input])
        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        self.critic = Model(inputs=[self.action_input, state_input], outputs=x)
        print("\n\n[DDPG] Critic Network: ")
        print(self.critic.summary())
