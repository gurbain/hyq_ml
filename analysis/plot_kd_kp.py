import os
import pickle

FILENAME = "/home/gurbain/results.pkl"

if os.path.exists(FILENAME):
    data = pickle.load(open(FILENAME, "rb"))
    print data


# X total pour different Kd (avec Kd en absices)

# X speed avec x/t_sim - t_trot