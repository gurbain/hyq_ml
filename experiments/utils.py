"""
Useful functions and classes to run experiments
"""

import os
import random
import sys
import string
import time


def signal_handler(sig, frame):

    print('\n\n--- User has quitted the manager! ---')
    sys.exit(0)


def timestamp():

    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def mkdir(path):

    if not os.path.exists(path):
        os.makedirs(path)


def gen_hash(to_exclude, n=12):

    s = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + \
                              string.digits) for _ in range(n))
    if s in to_exclude:
        s = gen_hash(to_exclude=to_exclude, n=n)
    return s


def get_parent(path):

    return os.path.abspath(os.path.join(path, os.pardir))