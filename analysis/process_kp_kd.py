import ConfigParser
import numpy as np
import os
import pickle
from tqdm import tqdm


EXP_FOLDER = "/home/gurbain/docker_sim/experiments/20181221-154105"
DATA_FILE_NAME = "data_kd_kp.pkl"


def browse_folder():

    data_filename = os.path.join(EXP_FOLDER, DATA_FILE_NAME)
    data = []

    for subdir in tqdm(os.listdir(EXP_FOLDER)):

        config_filename = os.path.join(os.path.join(EXP_FOLDER, subdir), "config.txt")
        physics_filename = os.path.join(os.path.join(EXP_FOLDER, subdir), "physics.pkl")
        if os.path.isfile(config_filename) and os.path.isfile(physics_filename):

                config = ConfigParser.ConfigParser()
                config.read(config_filename)
                init_impedance = eval(config.get("Physics", "init_impedance"))
                kp = init_impedance[0]
                kd = init_impedance[1]

                physics = pickle.load(open(physics_filename, "rb"))
                t_sim = physics["t_sim"][-1]
                t_trot = physics["t_trot"]
                t_real = physics["t_real"][-1]
                x = physics["x"][-1]
                y = physics["y"][-1]
                z = physics["z"][-1]

                data_it = {"t_sim": t_sim, "t_real": t_real, "t_trot": t_trot,
                           "Kp": kp, "Kd": kd, "x": x, "y": y, "z": z}

                data.append(data_it)

    pickle.dump(data, open(data_filename, "wb"), protocol=2)



if __name__ == "__main__":

    browse_folder()