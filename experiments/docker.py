import ConfigParser
import copy
import datetime
import dateutil.parser
import docker
import os
from picker import *
import random
import string
import time
import shutil
import signal
import sys
import subprocess
import sys
import threading


IMAGE = "hyq:latest"
IDLE_CMD = ""
RUN_CMD = "/bin/bash -c 'source /opt/ros/dls-distro/setup.bash;" + \
          "cd /home/gurbain/hyq_ml/hyq; roscore >> /dev/null 2>&1 & " \
          "timeout 10m python simulation.py "
IDLE_TASK = ["sleep", "infinity"]
RUN_TASK = [""]

SAVE_FOLDER = "/home/gurbain/docker_sim/"
TEST_SIM_CONFIG = "/home/gurbain/hyq_ml/config/sim_config_default.txt"
MOUNT_FOLDER = "/home/gurbain/docker_sim/"
MOUNT_OPT = "rw"


class Run(object):

    def __init__(self):

        print "this implements a run"

    def setup(self):
        return

    def save(self):
        return

    def fit(self):
        return

    def load(self):
        return

    def replay(self):
        return


class Cluster(object):

    def __init__(self, n_proc=4, verbose=True, logging=True):

        self.n_proc = n_proc
        self.verbose = verbose
        self.logging = logging

        self.engine = docker.from_env()
        self.img = IMAGE
        self.cmd = RUN_CMD
        self.task = RUN_TASK

        self.finished = False
        self.task_dirs = None
        self.exp_dir = None
        self.proc_count = 0
        self.cont_log_threads = []

        self.folder = SAVE_FOLDER
        self.mount_dir = MOUNT_FOLDER
        self.mount_opt = MOUNT_OPT

    def process(self, config_list):

        print "\n ----------------------------"
        print " --- Start New Experiment ---"
        print " ----------------------------\n"

        # Create experiment dir
        exp_root_folder = self.folder + "experiments/"
        utils.mkdir(exp_root_folder)
        self.exp_dir = exp_root_folder + util.timestamp()
        self.task_dirs = self.__create_task_folders(self.exp_dir, config_list)

        # Launch all experiments sequentially
        for i, c in enumerate(self.task_dirs):

            if self.proc_count >= self.n_proc:
                while self.__has_living_threads():
                    try:
                        self.__join_all_proc(1)
                    except KeyboardInterrupt:
                        self.finished = True
                        break

                self.proc_count = 0

            try:
                if self.finished:
                    break

                self.__start_proc(c, i)
                self.proc_count += 1

            except KeyboardInterrupt:
                self.finished = True

        # Finishing
        if not self.finished:
            while self.__has_living_threads():
                try:
                    self.__join_all_proc(1)
                except KeyboardInterrupt:
                    self.finished = True
        print "\n\n ------------------------------------------------" \
              "----------------------------------------------------------"
        print " --- Experiment is finished! Results are placed in " + str(self.exp_dir) + " ---"
        print " -----------------------------------------------------" \
              "-----------------------------------------------------\n"

        return self.exp_dir

    def __curr_date(self):

        return datetime.datetime.now().strftime('%d/%m %H:%M:%S')

    def __start_proc(self, folder, i):

        mounts = {self.mount_dir: {'bind': self.folder, 'mode': self.mount_opt}}
        try:
            cont = self.engine.containers.run(image=self.img, volumes=mounts, tty=True, remove=True,
                                              detach=True, command=self.cmd + str(folder) + "'")
        except KeyboardInterrupt:
            # The asynchronous function create the container but do not get the python object
            # We should kill all otherwise the last created container can be still up
            self.finished = True
            print "\n --- Please, never stop when docker jobs are starting! Must kill all, losing the logs!"
            while len(self.engine.containers.list()) > 0:
                try:
                    c = self.engine.containers.list()[-1]
                    c.stop(timeout=4)
                    c.kill()
                    c.remove(force=True)
                except docker.errors.APIError:
                    pass
                time.sleep(1)
            return

        print "\n -------------------------------------------------------------------------" \
              "--------------------"
        print " --- " + self.__curr_date() + \
              " | Sim Docker " + str(i + 1) + "/" + \
              str(len(self.task_dirs)) + " started with folder " + str(folder.split("/")[-1]) + \
              " and id " + str(cont.id)[:12] + " ---"
        print " ----------------------------------------------------------------------------" \
              "-----------------\n"

        self.cont_log_threads.append(threading.Thread(target=self.__stream_log, args=(cont, self.proc_count, folder,)))
        self.cont_log_threads[-1].start()

    def __join_all_proc(self, timeout=1):

        for c in self.cont_log_threads:
            if c is not None and c.isAlive():
                c.join(timeout)

    def __stream_log(self, container, num, folder):

        num_str = str(num + 1) + "/" + str(self.n_proc)
        id_str = str(container.id)[:12]
        log = ""
        first = True
        try:
            for l in container.logs(stream=True):

                # Handle stop
                if self.finished:
                    try:
                        container.stop(timeout=4)
                        container.kill()
                        container.remove(force=True)
                    except docker.errors.APIError:
                        pass
                    print num_str + " | " + id_str + " | " + \
                          self.__curr_date() + " | Exited or Killed!"
                    break

                # Print and/or save stdout
                if self.verbose:
                    if first:
                        sys.stdout.write("\n" + num_str + " | " + id_str + " | " +
                                         self.__curr_date() + " | " + l)
                        first = False
                        continue
                    sys.stdout.write(l.replace("\n", "\n" + num_str + " | " +
                                                     id_str + " | " + self.__curr_date() +
                                                     " | "))
                if self.logging:
                    log += l
        except ConnectionError:
            try:
                container.stop(timeout=4)
                container.kill()
                container.remove(force=True)
            except docker.errors.APIError:
                pass
            print "\n" + num_str + " | " + id_str + " | " + \
                  self.__curr_date() + " | Encountered error!"
            pass

        # Save the full log
        if self.logging:
            with open(folder + "/log.txt", 'w') as file:
                file.write(log)
                print "\r" + num_str + " | " + id_str + " | " + \
                      self.__curr_date() + " | Log in " + folder + "/log.txt"

    def __create_task_folders(self, root_folder, config):

        utils.mkdir(root_folder)
        liste = []
        for c in config:
            hashe = os.path.join(root_folder, gen_hash([o for o in os.listdir(root_folder)
                                                        if os.path.isdir(os.path.join(root_folder,o))]))
            liste.append(hashe)
            utils.mkdir(hashe)

            # Add a config file
            with open(hashe + '/config.txt', 'w') as configfile:
                c.write(configfile)

        return liste

    def __has_living_threads(self):

        return True in [t.isAlive() for t in self.cont_log_threads]
