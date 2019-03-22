from rl.callbacks import Callback
import numpy as np
import timeit
import rospy
import pandas as pd


class DataLogger(Callback):

    def __init__(self, filepath, interval=100):

        self.filepath = filepath + "/"
        self.interval = interval

        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dict that maps from episode to metrics array.
        # from each other.

        #gather data over serveral episodes
        self.data = {}

        #gather data over episode
        self.episode_time = {}
        self.step_time_real = {}
        self.step_time_sim = {}
        self.start_step_time_real = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.episode_time = {}
        self.info = {}
        self.start_episode = {}
        self.start_step_real = {}
        self.start_step_sim = {}

        self.step = 0

    def on_train_begin(self, logs):

        self.metrics_names = self.model.metrics_names

    def on_train_end(self, logs):

        self.save_data()

    def on_episode_begin(self, episode, logs):

        assert episode not in self.metrics
        assert episode not in self.start_episode

        self.step_time_real[episode] = []
        self.step_time_sim[episode] = []
        self.start_step_time_real[episode] = []
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.start_episode[episode] = timeit.default_timer()
        self.metrics[episode] = []
        self.info[episode] = []

    def on_episode_end(self, episode, logs):

        self.episode_time[episode] = timeit.default_timer() - self.start_episode[episode]

        self.data[episode] = {"episode": episode,
                              "episode_time": self.episode_time[episode],
                              "start_step_sim":self.start_step_time_real[episode],
                              "step_time_real": self.step_time_real[episode],
                              "step_time_sim": self.step_time_sim[episode],
                              "observations": self.observations[episode],
                              "rewards": self.rewards[episode],
                              "actions": self.actions[episode],
                              "start_episode": self.start_episode[episode],
                              "metrics": self.metrics[episode],
                              "info": self.info[episode]}

        if self.interval is not None and episode % self.interval == 0:
            self.save_data()

        # Clean up.
        del self.step_time_real[episode]
        del self.step_time_sim[episode]
        del self.start_step_time_real[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.start_episode[episode]
        del self.metrics[episode]
        del self.info[episode]

    def on_step_begin(self, step, logs):

        self.start_step_real[step] = timeit.default_timer()
        self.start_step_sim[step] = rospy.get_time()

    def on_step_end(self, step, logs):

        episode = logs['episode']
        duration_step_sim = rospy.get_time() - self.start_step_sim[step]
        duration_step_real = timeit.default_timer() - self.start_step_real[step]

        self.step_time_real[episode].append(duration_step_real)
        self.step_time_sim[episode].append(duration_step_sim)
        self.start_step_time_real[episode].append(self.start_step_sim[step])
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.info[episode].append(logs['info'])

        self.step += 1

        del self.start_step_sim[step]
        del self.start_step_real[step]

    def save_data(self):

        if len(self.data.keys()) == 0:
            return

        episodes_keys = self.data.keys()

        columns = self.data[episodes_keys[0]].keys()
        df = pd.DataFrame(columns=columns)
        for episodes_key in episodes_keys:
            df = df.append(self.data[episodes_key], ignore_index=True)
            del self.data[episodes_key]

        path = self.filepath + "data_episode_{}_{}.p".format(np.min(episodes_keys), np.max(episodes_keys))

        print('Save data: {}'.format(path))
        df.to_pickle(path)
        del df
