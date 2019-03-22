"""
The main framework to start a simulation HyQ Gym + Learning Agent
"""

import ConfigParser
from hyq_agents import *
import hyq_gym
import gym
import numpy as np
import sys

import utils


class Sim(object):

    def __init__(self, sim_folder):

        self.sim_folder = sim_folder
        self.config_file = sim_folder + "/config.txt"

        self.config = None
        self.agent = None
        self.env = None

        self.create()

    def create(self):

        # Get the config
        config_parser = ConfigParser.ConfigParser()
        config_parser.read(self.config_file)
        self.config = {s: dict(config_parser.items(s)) for s in config_parser.sections()}

        # Set-up Gym Environment
        self.env = gym.make(self.config["Gym"]["env_name"])
        np.random.seed(int(self.config["Gym"]["seed"]))
        self.env.seed(int(self.config["Gym"]["seed"]))

        # Create agent depending on type
        self.config["Agent"]["sim_folder"] = self.sim_folder
        self.config["Agent"]["env_name"] = self.config["Gym"]["env_name"]
        self.agent = eval(self.config["Agent"]["agent"].upper())(self.env, self.config["Agent"])

    def fit(self):

        self.agent.fit()

    def test(self):

        self.agent.test()


if __name__ == "__main__":

    if len(sys.argv) == 2:
        s = Sim(str(sys.argv[1]))
        s.fit()

    elif len(sys.argv) == 3:
        if str(sys.argv[1]) == "train":
            s = Sim(str(sys.argv[2]))
            s.fit()
        elif str(sys.argv[1]) == "test":
            s = Sim(str(sys.argv[2]))
            s.test()
        else:
            print "[HyQ Exp] Unknown command-line parameters"

    else:
        print "[HyQ Exp] Unknown command-line parameters"
