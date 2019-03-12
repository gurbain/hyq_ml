import json

class ConfigExperiment():
    def __init__(self):
        self.config = {}
        self.set_default()

    def set_default(self):
        # training vars
        self.config['ENV_NAME'] = 'tigrillo-v25'

        # agent vars
        self.config['AGENT'] = 'DDPG'
        self.config['MODEL_TYPE'] = 'basic'

        self.config['nb_steps'] = 10000000
        self.config['nb_steps_warmup_critic'] = 1000
        self.config['nb_steps_warmup_actor'] = 1000
        self.config['gamma'] = 0.99
        self.config['target_model_update'] = 1e-3
        self.config['lr1'] = 1e-4
        self.config['lr2'] = 1e-3

        #observation normalizer
        self.config['WhiteningNormalizerProcessor'] = False

        #memory vars
        self.config['MEMORY_LIMIT'] = 100000
        self.config['MEMORY_WINDOW'] = 1

        #Random procceses
        self.config['RANDOM_PROCCES'] = 'OrnsteinUhlenbeckProcess'
        self.config['theta'] = .15
        self.config['mu'] = 0.
        self.config['sigma'] = 0.1

        self.print_config()


    def save_config_json(self,path):
        with open(path, 'w') as fp:
            json.dump(self.config, fp)

    def load_config_json(self,path):
        with open(path, 'r') as fp:
            self.config = json.load(fp)

    def print_config(self):
        print(self.config)

    def get_var(self,name):
        try:
            return(self.config[name])
        except Exception as e:
            print(e)

    def set_var(self,name,var):
        self.config[name] = var