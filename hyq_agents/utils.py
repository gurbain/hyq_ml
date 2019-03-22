"""
Useful functions and classes to run DDPG agent
"""

import os
import time


def timestamp():
    """ Return a string stamp with current date and time """

    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def mkdir(path):
    """ Create a directory if it does not exist yet """

    if not os.path.exists(path):
        os.makedirs(path)


def is_file(path):
    """ Check if file exists """

    return os.path.isfile(path)
