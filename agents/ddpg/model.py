from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate


class ModelGenerator():

    def create(self, env, model_type):

        if(model_type == 'basic'):
            return self.create_basic(env)

        elif(model_type == 'basic_3_layer'):
            return self.create_3_layers(env)

        else:
            raise Exception('ModelError', 'Model \'' + str(model_type) + '\' not found')

    def create_basic(self, env):

        assert len(env.action_space.shape) == 1
        nb_actions = env.action_space.shape[0]

        actor = Sequential()
        actor.add(Flatten(input_shape=(1, 1,) + env.observation_space.shape))
        actor.add(Dense(64))
        actor.add(Activation('relu'))
        actor.add(Dense(32))
        actor.add(Activation('relu'))
        actor.add(Dense(nb_actions))
        actor.add(Activation('linear'))
        print("\n\n[DDPG] Actor Network: ")
        print(actor.summary())

        action_input = Input(shape=(nb_actions,), name='action_input')
        state_input = Input(shape=(1, 1,) + env.observation_space.shape, name='state_input')
        flattened_state = Flatten()(state_input)
        x = Dense(64)(flattened_state)
        x = Activation('relu')(x)
        x = Concatenate()([x, action_input])
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, state_input], outputs=x)
        print("\n\n[DDPG] Critic Network: ")
        print(critic.summary())

        return actor, critic, action_input

    def create_3_layers(self,env):
        assert len(env.action_space.shape) == 1
        nb_actions = env.action_space.shape[0]

        # Next, we build a very simple model.
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        actor.add(Dense(128))
        actor.add(Activation('relu'))
        actor.add(Dense(64))
        actor.add(Activation('relu'))
        actor.add(Dense(32))
        actor.add(Activation('relu'))
        actor.add(Dense(nb_actions))
        actor.add(Activation('tanh'))
        print("\n\n[DDPG] Actor Network: ")
        print(actor.summary())

        action_input = Input(shape=(nb_actions,), name='action_input')
        state_input = Input(shape=(1,) + env.observation_space.shape, name='state_input')
        flattened_state = Flatten()(state_input)
        x = Dense(128)(flattened_state)
        x = Activation('relu')(x)
        x = Concatenate()([x, action_input])
        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, state_input], outputs=x)
        print("\n\n[DDPG] Critic Network: ")
        print(critic.summary())

        return actor, critic, action_input