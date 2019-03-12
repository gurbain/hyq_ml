from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate

class ModelZoo():

    def create_model(self,env,MODEL_TYPE):
        print(MODEL_TYPE)
        if(MODEL_TYPE == 'basic'):
            return self.create_model_basic(env)
        elif(MODEL_TYPE == 'basic_3_layer'):
            return self.create_model_3_layer(env)
        else:
            raise Exception('modelerror', 'defined model not found')

    def create_model_basic(self,env):
        assert len(env.action_space.shape) == 1
        nb_actions = env.action_space.shape[0]

        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        actor.add(Dense(400))
        actor.add(Activation('relu'))
        actor.add(Dense(300))
        actor.add(Activation('relu'))
        actor.add(Dense(nb_actions))
        actor.add(Activation('tanh'))
        print(actor.summary())

        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Dense(400)(flattened_observation)
        x = Activation('relu')(x)
        x = Concatenate()([x, action_input])
        x = Dense(300)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        print(critic.summary())

        return actor, critic, action_input

    def create_model_3_layer(self,env):
        assert len(env.action_space.shape) == 1
        nb_actions = env.action_space.shape[0]

        # Next, we build a very simple model.
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        actor.add(Dense(1024))
        actor.add(Activation('relu'))
        actor.add(Dense(512))
        actor.add(Activation('relu'))
        actor.add(Dense(256))
        actor.add(Activation('relu'))
        actor.add(Dense(nb_actions))
        actor.add(Activation('tanh'))
        print(actor.summary())

        action_input = Input(shape=(nb_actions,), name='action_input')
        observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Dense(1024)(flattened_observation)
        x = Activation('relu')(x)
        x = Concatenate()([x, action_input])
        x = Dense(512)(x)
        x = Activation('relu')(x)
        x = Dense(256)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        print(critic.summary())
        return actor, critic, action_input