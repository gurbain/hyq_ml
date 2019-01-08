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


# Edit the following variables depending on your cluster configuration
# And check that every computer can communicate via SSH without user password
# (SSH keys with no password)
USER = "gurbain"
HOST = "paard"
HOST_IP = '192.168.0.2'
HOST_PORT = '5000'
MACHINES = ["paard",  "geit", "kat", "schaap"]
KEY_FILE = "/home/gurbain/.docker_swarm_key"

JOB_LIMIT = 200
IMAGE = "hyq:latest"
IDLE_CMD = ""
RUN_CMD = "/bin/bash -c 'source /opt/ros/dls-distro/setup.bash;" + \
          "cd /home/gurbain/hyq_ml/hyq; roscore & timeout 5m python simulation.py "
IDLE_TASK = ["sleep", "infinity"]
RUN_TASK = [""]

SAVE_FOLDER = "/home/gurbain/docker_sim/"
TEST_SIM_CONFIG = "/home/gurbain/hyq_ml/config/sim_config_default.txt"
MOUNT_FOLDER = "/home/gurbain/docker_sim/"
MOUNT_OPT = "rw"


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
        s = generate_random(to_exclude=to_exclude, n=n)
    return s

class Lan(object):

    def __init__(self, engine):

        self.engine = engine

        self.swarm_user = USER
        self.swarm_host = HOST
        self.swarm_host_ip = HOST_IP
        self.swarm_host_port = HOST_PORT
        self.swarm_machines = MACHINES
        self.swarm_key_file = KEY_FILE

        self.manager_init = False
        self.key = None

    def change(self):

        machines_on = []
        for m in self.engine.nodes.list():
            if str(m.attrs["Status"]["State"]) == "ready":
                machines_on.append(str(m.attrs["Description"]["Hostname"].split(".")[0]))
        machines_mask = [m in machines_on for m in self.swarm_machines]
        machines_ind = Picker(title="Select the computers to enable in the swarm",
                              options=self.swarm_machines,
                              init_options=machines_mask).getIndex()
        machines_new = [self.swarm_machines[ind] for ind in machines_ind]

        # check for master
        if self.swarm_host not in machines_new:
            if len(machines_new) != 0:
                print "\n--- To make the swarm works, you need to keep the manager " +  \
                      str(self.swarm_host) + " ---\n"
            else:
                self.stop()
            return

        # Add new machines
        for m in machines_new:
            if m not in machines_on:
                self.add(m)

        # Remove machines
        for m in machines_on:
            if m  not in machines_new:
                self.rm(m)

    def start(self):

        print "\n--- Starting swarm manager on this computer ---\n"
        r = self.engine.swarm.init(advertise_addr=self.swarm_host_ip, listen_addr='0.0.0.0:5000',
                                   force_new_cluster=False, snapshot_interval=5000,
                                   log_entries_for_slow_followers=1200)
        self.key = self.engine.swarm.attrs["JoinTokens"]["Worker"]
        self.__write_key(self.swarm_host)

    def stop(self):

        print "\n--- Stoping the swarm on this computer ---\n"
        machines_on = [str(m.attrs["Description"]["Hostname"].split(".")[0]) for m in self.engine.nodes.list()]

        for m in machines_on:
            if m != self.swarm_host:
                self.__leave_swarm(m)

        self.engine.swarm.leave(force=True)

    def add(self, machine):

        self.__write_key(machine)
        self.__join_swarm(machine)

    def rm(self, machine):

        self.__leave_swarm(machine)

    def is_init(self):
        
        init = True
        try:
            m = self.engine.nodes.list()
        except docker.errors.APIError as e:
            if e.status_code == 503:
                init = False

        return init

    def __join_swarm(self, machine):

        print "\n--- Node " + str(machine) + " is joing swarm ---\n"
        cmd = ['ssh', self.swarm_user + '@' + machine, 
               "docker swarm join --token $(<" + str(self.swarm_key_file) + \
               ") " + str(self.swarm_host_ip) + ":" + str(self.swarm_host_port)]

        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        p.communicate()

    def __leave_swarm(self, machine):

        print "\n--- Node " + str(machine) + " is leaving swarm ---\n"
        cmd = ['ssh', self.swarm_user + '@' + machine, "docker swarm leave --force"]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        p.communicate()

    def __read_swarm_key(self):

        if self.key == None:
            with open(self.swarm_key_file, 'r') as file:
                self.key = file.read()

    def __write_key(self, machine):

        self.__read_swarm_key()
        cmd = ['ssh', self.swarm_user + '@' + machine, ' cat - > ' + self.swarm_key_file]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        p.stdin.write(self.key)
        p.communicate()


class Services(object):

    def __init__(self, engine):

        self.engine = engine

        self.srv_img = IMAGE
        self.srv_idle_cmd = IDLE_CMD
        self.srv_run_cmd = RUN_CMD
        self.srv_idle_task = IDLE_TASK
        self.srv_run_task = RUN_TASK
        self.swarm_machines = MACHINES
        self.srv_limit = JOB_LIMIT

        self.def_name = 'sim_wrk_'

        self.nodes_id = []
        self.nodes_name = []  
        for m in self.engine.nodes.list():
            if str(m.attrs["Status"]["State"]) == "ready":
                self.nodes_id.append(m.id)
                self.nodes_name.append(str(m.attrs["Description"]["Hostname"].split(".")[0]))

        self.srv_names = [str(s.name) for s in self.engine.services.list()]
        self.srv_ids = [str(s.id) for s in self.engine.services.list()]

        self.workers  = []

    def browse(self):

        self.nodes_name = []  
        for m in self.engine.nodes.list():
            if str(m.attrs["Status"]["State"]) == "ready":
                self.nodes_id.append(m.id)
                self.nodes_name.append(str(m.attrs["Description"]["Hostname"].split(".")[0]))

        self.srv_names = [str(s.name) for s in self.engine.services.list()]
        self.srv_ids = [str(s.id) for s in self.engine.services.list()]
        self.workers  = []

        for i, srv in enumerate(self.srv_ids):
            try:
                s = self.engine.services.get(srv)
                t = None
                for t2 in s.tasks():
                    if t2["Spec"]["ContainerSpec"]["Image"] == self.srv_img:
                        if t is None:
                            t = t2
                        elif dateutil.parser.parse(t2["CreatedAt"]) > dateutil.parser.parse(t["CreatedAt"]):
                            t = t2
                if t is not None:
                    try:
                        host = self.engine.nodes.get(t["NodeID"]).attrs["Description"]["Hostname"].split(".")[0]
                        status = t["Status"]["State"]
                        # TODO: print the previous log if it fails
                        # if "UpdatedAt" in  t.keys() and status not in ["failed", "shutdown"]:
                        #     creation = int((dateutil.parser.parse(t["UpdatedAt"]) + datetime.timedelta(hours=1)).strftime("%s"))
                        # else:
                        creation = int((dateutil.parser.parse(t["CreatedAt"]) + datetime.timedelta(hours=1)).strftime("%s"))
                        service_name = s.name
                        version = s.version
                        args = ""
                        if 'Args' in t["Spec"]["ContainerSpec"].keys():
                            args += " ".join(t["Spec"]["ContainerSpec"]["Args"])
                        if args == " ".join(self.srv_idle_task):
                            cmd_type = "idle"
                        else:
                            cmd_type = "run"

                        # print cmd, status, service_name
                        self.workers.append({"type": cmd_type, "service_version": version,
                                             "service_id": srv, "service_name": service_name,
                                             "host": host, "cmd": args, "status": status,
                                             "last_timestamp": creation})

                    except (ValueError, docker.errors.NotFound, KeyError):
                        pass
            except docker.errors.NotFound:
                pass

    def status(self):

        self.browse()

        # Print summary
        wrk_idle = [w for w in self.workers if w["type"] == "idle" ]
        wrk_run = [w for w in self.workers 
                   if (w["type"] == "run" and w["status"] in ["ready", "starting", "running"])]
        wrk_finished =  [w for w in self.workers
                         if (w["type"] == "run" and w["status"] in ["complete", "shutdown"])]
        wrk_failed = [w for w in self.workers if (w["type"] == "run" and w["status"] == "failed")]

        print "\n--- IDLE Services: " + str(len(wrk_idle)) + " ---"
        for i, n in enumerate(self.nodes_name):
            print str(n) + ": " + str(len([w for w in wrk_idle if w["host"] == n]))
        print "\n--- RUNNING Services: " + str(len(wrk_run)) + " ---"
        for i, n in enumerate(self.nodes_name):
            print str(n) + ": " + str(len([w for w in wrk_run if w["host"] == n]))
        print "\n--- FINISHED Services: " + str(len(wrk_finished)) + " ---"
        for i, n in enumerate(self.nodes_name):
            print str(n) + ": " + str(len([w for w in wrk_finished if w["host"] == n]))
        print "\n--- FAILED Services: " + str(len(wrk_failed)) + " ---"
        for i, n in enumerate(self.nodes_name):
            print str(n) + ": " + str(len([w for w in wrk_failed if w["host"] == n]))

    def rm(self):

        self.browse()
        srv_name = [w["service_name"] for w in self.workers]
        srv_id = [w["service_id"] for w in self.workers]
        indexes = Picker(title="Select the services to kill and remove", 
                         options=filter(lambda k: self.def_name in k, srv_name)).getIndex()
        if indexes == []:
            print "No selection. Aborded!"
        else:
            for i in indexes:
                print "\n--- Killing and deleting service \"" + str(srv_name[i]) + "\" ---\n"
                ids = self.engine.services.get(srv_id[i])
                ids.remove()

    def add(self, num, tot=None):

        if tot is None:
            self.browse()
            tot = len(self.workers)

        if int(num) + tot > self.srv_limit:
                print "\nYou cannot exceed the limit of " + str(self.srv_limit) + " in total!"
                return 
                
        for i in range(int(num)):
            srv_replicas_mode = docker.types.ServiceMode("replicated", replicas=1)
            srv_restart_policy =  docker.types.RestartPolicy(condition='none', 
                                                             delay=0, max_attempts=0, 
                                                             window=0) 
            name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(3))
            srv_id = self.engine.services.create(image=self.srv_img, command=self.srv_idle_cmd,
                                                 args=self.srv_idle_task, 
                                                 name=self.def_name + name, mode=srv_replicas_mode,
                                                 restart_policy=srv_restart_policy)


class Tasks(object):

    def __init__(self, engine=None):

        if engine is None:
            self.engine = docker.from_env()
        else:
            self.engine = engine
        self.cluster = Services(self.engine)

        self.srv_status = None

        self.running = []
        self.task_dirs = None
        self.exp_dir = None
        self.res_dirs = []

        self.folder = SAVE_FOLDER
        self.mount_dir = MOUNT_FOLDER
        self.mount_opt = MOUNT_OPT

        self.max_job_attemps = 5
        self.max_waiting_time = 60
        self.max_time_update = 20

        self.docker_log = ""

    def process(self, config_list):

        print "\n--- Start New Experiment ---\n"
        self.docker_log = ""

        # Check if another simulation is running
        self.__check_status()

        # Create experiment dir
        exp_root_folder = self.folder + "experiments/"
        mkdir(exp_root_folder)
        self.exp_dir = exp_root_folder + timestamp()
        self.task_dirs = self.__create_task_folders(self.exp_dir, config_list)

        # Check status first before starting experiment
        self.cluster.browse()
        self.srv_status = copy.deepcopy(self.cluster.workers)

        # Launch all experiments
        while len(self.task_dirs) != 0:

            self.__monitor()

            n_idle_jobs = len([w for w in self.cluster.workers 
                               if (w["type"] == "idle" and w["status"] == "running")])
            n_desired_jobs = len(self.task_dirs)

            if n_idle_jobs > 0:
                print "--- Running " + str(min(n_desired_jobs, n_idle_jobs)) + " new jobs. " + \
                    "Still " + str(max(0, n_desired_jobs - n_idle_jobs)) + " in the queue ---\n"

                for _ in range(min(n_desired_jobs, n_idle_jobs)):
                    self.res_dirs.append(self.task_dirs.pop(0))
                    self.__start_job(self.res_dirs[-1])
                if min(n_desired_jobs, n_idle_jobs) > 0:
                    time.sleep(1)

        # Wait for all results
        while len(self.running) != 0:
            self.__monitor()

        print "--- Experiment is finished. Results are placed in " + str(self.exp_dir) + " ---\n"
        self.__monitor()
        self.__save_docker_log()
        return self.res_dirs

    def logs(self):

        self.cluster.browse()

        worker_run_name = [w["service_name"] for w in self.cluster.workers if w["type"] == "run"]
        worker_run_id = [w["service_id"] for w in self.cluster.workers if w["type"] == "run"]
        worker_run_creat = [w["last_timestamp"] for w in self.cluster.workers if w["type"] == "run"]
        indexes = Picker(title="Select the service to display the logs", 
                         options=worker_run_name).getIndex()
        if indexes == []:
            print "No selection. Aborded!"
        else:
            for i in indexes:
                print "\n--- Printing logs from service \"" + str(worker_run_name[i]) + "\" ---\n"
                try:
                    with utils.Timeout(10):
                        ids = self.engine.services.get(worker_run_id[i])
                        logs = ids.logs(details=False, stdout=True, stderr=True,
                                    timestamps=False, since=worker_run_creat[i])
                        for l in logs:
                            sys.stdout.write(l)

                except Timeout.Timeout as e: 
                    self.self.docker_log += str(utils.timestamp()) + "| ERROR | "  + str(e)
                    pass

    def __monitor(self):

        self.cluster.browse()

        cluster_save = copy.deepcopy(self.cluster.workers)
        # print self.srv_status, self.cluster.workers

        if self.srv_status is not None:

             # Check if any job failed
            for new in cluster_save:
                if new["status"] in ["rejected", "failed", "pause"] and new["type"] == "run":

                    # Save the error log
                    s_id = new["service_id"]
                    if s_id in [r["service_id"] for r in self.running]:
                        r = self.running[[r["service_id"] for r in self.running].index(s_id)]
                        folder = r["folder"]
                        try:
                            with utils.Timeout(10):
                                with open(folder + "/error_log.txt", 'w') as file:
                                    ids = self.engine.services.get(s_id)
                                    logs = ids.logs(details=False, stdout=True, stderr=True,
                                            timestamps=False, since=new["last_timestamp"])
                                    for l in logs:
                                        file.write(l)
                        except Timeout.Timeout as e: 
                            self.self.docker_log += str(utils.timestamp()) + "| ERROR | "  + str(e)
                            pass
                        
                    # Set the service in idle mode
                    self.__start_idle(new, remove=False)

                    # restart the job
                    r_id = [r["service_id"] for r in self.running]
                    if new["service_id"] in r_id:

                        # Set to idle
                        folder = self.running[r_id.index(new["service_id"])]["folder"]
                        print "--- Failed job with folder " + str(folder.split("/")[-1]) + \
                              " is re-started on service " + str(new["service_name"]) + " ---\n"

                        # Restart
                        self.__start_job(folder, append=False)
                    else:
                        print "--- An old failed job is stopped on service " + \
                              str(new["service_name"]) + " ---\n"

                if new["status"] in ["shutdown", "rejected"] and new["type"] == "idle":
                    print "--- Failed idle job is restarted on service " + \
                           str(new["service_name"]) + " ---\n"
                    self.__start_idle(new)

            # Check if status changed from running to complete
            for new in cluster_save:
                if new["status"] == "complete":
                    running_services = [r["service_id"] for r in self.running \
                                        if r["status"] in ["started", "restarted"]]
                    if new["service_id"] in running_services:
                            self.running[[r["service_id"] for r in self.running].index(new["service_id"])]["status"] = "saving"
                            old = self.srv_status[[s["service_id"] for s in self.srv_status].index(new["service_id"])]
                            self.__modif_callback(new, old)
            # for old in self.srv_status:
            #     for new in cluster_save:

            #         if old["service_name"] == new["service_name"]:
            #             if new["status"] == "complete":
            #                 if old["status"] != "complete" or old["service_version"] != new["service_version"]:
            #                     self.__modif_callback(new, old)

        self.srv_status = cluster_save

    def __check_status(self):

        self.cluster.browse()

        if len([w for w in self.cluster.workers if w["type"] == "run"]) > 0:

            try:
                to_add = raw_input("There are already running jobs. Would you like to kill them first? (Y/n/q): ")
                if to_add in ["", "Y", "y"]:
                    self.__start_idle(srv=None, remove=False)
                    time.sleep(2)
                elif to_add in ["q", "Q"]:
                    exit(1)
            except (ValueError, TypeError, NameError, SyntaxError):
                exit(1)

        print ""

    def __start_job(self, folder, append=True):

        # Wait for a service to be available
        self.cluster.browse()
        t_init = time.time()
        while len([w["service_id"] for w in self.cluster.workers 
                   if (w["type"] == "idle" and w["status"] == "running")]) == 0:
            self.cluster.browse()
            if time.time() - t_init > self.max_waiting_time:
                print "--- ERROR: Timeout when waiting for idle service ---\n"
                self.__save_docker_log()
                exit(-1)

        # Pick the first idle service available in the list
        srv_ids = random.choice([w["service_id"] for w in self.cluster.workers 
                   if (w["type"] == "idle" and w["status"] == "running")])
        try:
            srv = self.engine.services.get(srv_ids)
        except docker.errors.NotFound:
            print "--- ERROR: Service does not exist ---\n"
            self.__save_docker_log()
            exit(-1)
        srv_name = srv.name
        print "--- A new job is started with folder " + str(folder.split("/")[-1]) + \
              " on service " + str(srv_name) + " ---\n"

        if append:
            self.running.append({"service_name": srv_name, "folder": folder, "launch": 1,
                                 "service_id": srv_ids, "time_init": time.time(),
                                 "status": "started"})
        else:
            if folder in [r["folder"] for r in self.running]:
                r = self.running[[r["folder"] for r in self.running].index(folder)]
                r["launch"] += 1
                r["time_init"] = time.time()
                r["service_name"] = srv_name
                r["service_id"] = srv_ids
                r["status"] = "restarted"
                
                if self.running[[r["folder"] for r in self.running].index(folder)]["launch"] > self.max_job_attemps:
                    print "--- ERROR: Timeout! You re-started the same job " + \
                          str(self.max_job_attemps) + " times without success! ---\n"
                    self.__save_docker_log()
                    exit(-1)
            else:
                print "--- ERROR: You cannot use relaunch a job that has not been launched yet! ---\n"
                self.__save_docker_log()
                exit(-1)

        # Update the service with the new command
        success = False
        t_init = time.time()
        while not success:
            try:
                srv.update(image=self.cluster.srv_img, 
                           command=self.cluster.srv_run_cmd + str(folder) + "'",
                           args=self.cluster.srv_run_task,
                           mounts=[self.folder + ":" + self.mount_dir + ":" + self.mount_opt])
                success = True
            except docker.errors.APIError as e:
                self.self.docker_log += str(utils.timestamp()) + "| ERROR | "  + str(e)
                pass

            if time.time() - t_init > self.max_time_update:
                print "--- ERROR: Timeout when trying to set service to running state ---\n"
                self.__save_docker_log()
                exit(-1)
        
    def __start_idle(self, srv=None, remove=True):

        self.cluster.browse()

        if srv is None:
            srv_ids = [w["service_id"] for w in self.cluster.workers if w["type"] == "run"]
        else:
            srv_ids = [srv["service_id"]]

        for srv_id in srv_ids:
            try:
                s = self.engine.services.get(srv_id)
            except docker.errors.NotFound:
                print "--- ERROR: Service does not exist ---\n"
                self.__save_docker_log()
                exit(-1)

            # Update the service with the new command
            success = False
            t_init = time.time()
            while not success:

                try:
                    s.update(image=self.cluster.srv_img, command=self.cluster.srv_idle_cmd,
                             args=self.cluster.srv_idle_task, mounts=[])
                    success = True
                except docker.errors.APIError as e:
                    self.self.docker_log += str(utils.timestamp()) + "| ERROR | "  + str(e)
                    pass

                if time.time() - t_init > self.max_time_update:
                    print "--- ERROR: Timeout when trying to set service to idle state ---\n"
                    self.__save_docker_log()
                    exit(-1)

            if remove:
                success = False
                for r in self.running:
                    if r["service_id"] == srv_id:
                        self.running.pop(self.running.index(r))
                        success = True
                if success == False:
                    print "--- ERROR: Cannot remove unexisting running service " + \
                          str(srv_id) + " ---\n"
                    self.__save_docker_log()
                    exit(-1)

    def __del_callback(self, srv_name):

        print "\n--- The service " + str(srv_name) + " has been deleted ---"
        # check results

    def __add_callback(self, srv_name):

        print "\n--- New service " + str(srv_name) + " has been added ---"

    def __modif_callback(self, srv_new, srv_old):

        print "--- Service " + str(srv_new["service_name"]) + " has change status from " + \
              str(srv_old["status"]) + " to " + str(srv_new["status"]) + " ---"

        # Check out results
        if srv_new["type"] == "run":
            print "\n--- Saving results for service " + str(srv_new["service_name"]) + " ---\n"

            s_id = srv_new["service_id"]
            r = self.running[[r["service_id"] for r in self.running].index(s_id)]
            folder = r["folder"]
            try:
                with utils.Timeout(10):
                    with open(folder + "/log.txt", 'w') as file:
                        ids = self.engine.services.get(s_id)
                        logs = ids.logs(details=False, stdout=True, stderr=True,
                                        timestamps=False, since=srv_old["last_timestamp"])
                        for l in logs:
                            file.write(l)
            except Timeout.Timeout as e: 
                self.self.docker_log += str(utils.timestamp()) + "| ERROR | "  + str(e)
                pass

        # Restart the node
        self.__start_idle(srv=srv_new)

    def __create_task_folders(self, root_folder, config):

        mkdir(root_folder)
        liste = []
        for c in config:
            hashe = os.path.join(root_folder, gen_hash([o for o in os.listdir(root_folder) 
                                                        if os.path.isdir(os.path.join(root_folder,o))]))
            liste.append(hashe)
            mkdir(hashe)

            # Add a config file
            with open(hashe + '/config.txt', 'w') as configfile:
                c.write(configfile)

        return liste

    def __save_docker_log(self):

        with open(self.exp_dir + '/docker_err_log.txt', 'w') as f:
            f.write(self.docker_log)


class Manager(object):

    def __init__(self):

        self.engine = docker.from_env()

    def process(self, arg_list):

        if arg_list[0] in ["swarm", "net", "cluster", "network", "lan"]:
            self.process_lan()

        elif arg_list[0] in ["srv", "wrk", "worker", "workers", "service", "services"]:
            self.process_srv(arg_list)

        elif arg_list[0] in ["jobs", "job", "sim", "simulation", "simulations", 
                             "sims", "compute", "physics"]:
            self.process_tsk(["job"] + arg_list[1:])

        elif arg_list[0] in ["log", "logs"]:
            self.process_tsk(["logs"] + arg_list[1:])

        elif arg_list[0] in ["task", "tasks"]:
            self.process_tsk(arg_list[1:])

        else:
            print "\n--- Invalid argument. Please check the file ---\n"

    def process_lan(self):

        lan = Lan(self.engine)

        if lan.is_init():
            lan.change()
        else:
            lan.start()

    def process_srv(self, arg_list):

        srv = Services(self.engine)

        if len(arg_list) == 1:
            srv.status()

        else:
            if arg_list[1] in ["add"]:
                if len(arg_list) <= 2:
                    print "\n--- Provide the number of services to add ---\n"
                else:
                    srv.add(arg_list[2])

            elif arg_list[1] in ["rm", "remove"]:
                srv.rm()

            elif arg_list[1] in ["status"]:
                srv.change()

            else:
                print "\n--- Invalid WRK argument. Please check the file ---\n"

    def process_tsk(self, arg_list):

        tsk = Tasks(self.engine)
        if arg_list[0] in ["test", "job", "jobs"]:
            if len(arg_list) > 1:
                config = ConfigParser.ConfigParser()
                config.read(TEST_SIM_CONFIG)
                config_list = [config for i in range(int(arg_list[1]))]
            else:
                config_list = ["default"]
            results = tsk.process(config_list)

        elif arg_list[0] in ["stop", "leave"]:
            tsk.stop()

        elif arg_list[0] in ["logs", "log"]:
            tsk.logs()

        else:
            print "\n--- Invalid TSK argument. Please check the file ---\n"


if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal_handler)

    if len(sys.argv) <= 1:
        print "\n--- This manager requires at least one argument (see file) ---\n"
        exit()

    m = Manager()
    m.process(sys.argv[1:])
