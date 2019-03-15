"""
Useful functions and classes to run DDPG agent
"""

import json
import os
import time


class Config():

    def __init__(self):

        self.config = {}
        self.set_default()

    def set_default(self):

        # Environment variables
        self.config['env_name'] = 'hyq-v2'

        # Neural Network variables
        self.config['model_type'] = 'basic'

        # Training variables
        self.config['agent'] = 'ddpg'
        self.config['nb_steps'] = 10000000
        self.config['nb_steps_warmup_critic'] = 1000
        self.config['nb_steps_warmup_actor'] = 1000
        self.config['gamma'] = 0.99
        self.config['target_model_update'] = 1e-3
        self.config['lr1'] = 1e-4
        self.config['lr2'] = 1e-3
        self.config['metrics'] = 'mae'
        self.config['memory_limit'] = 100000
        self.config['memory_window'] = 1
        self.config['random_theta'] = .15
        self.config['random_mu'] = 0.
        self.config['random_sigma'] = 0.1

        self.print_config()

    def save_config_json(self, path):

        with open(path, 'w') as fp:
            json.dump(self.config, fp)

    def load_config_json(self, path):
        
        with open(path, 'r') as fp:
            self.config = json.load(fp)

    def print_config(self):
        
        print("\n[DDPG] Experiment config: " + str(self.config))

    def get(self,name):
        
        try:
            return(self.config[name])
        except Exception as e:
            print("[DDPG] ", e)

    def set(self,name,var):
        
        self.config[name] = var

def timestamp():
    """ Return a string stamp with current date and time """

    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def mkdir(path):
    """ Create a directory if it does not exist yet """

    if not os.path.exists(path):
        os.makedirs(path)