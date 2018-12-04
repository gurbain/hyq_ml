from datetime import datetime
import dateutil.parser
import docker
from picker import *
import random
import string
import time
import sys
import subprocess

# Edit the following variables depending on your cluster configuration
# And check that every computer can communicate via SSH without user password
# (SSH keys with no password)
USER = "gurbain"
HOST = "paard"
HOST_IP = '172.18.20.20'
HOST_PORT = '5000'
MACHINES = ["paard", "hond", "geit", "kat", "koe", "schaap"]
KEY_FILE = "/home/gurbain/.docker_swarm_key"

JOB_LIMIT = 20
IMAGE = "hyq:latest"
IDLE_TASK = "tail -f /dev/null"
RUN_TASK = "ping -c 30 172.18.20.240"
           # source /opt/ros/dls-distro/setup.bash; \
           # export PATH=\"/home/gurbain/hyq_ml/docker/bin:$PATH\"; \
           # cd /home/gurbain/hyq_ml/hyq; \
           # roscore & python physics.py rcf'"


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
                machines_on.append(str(m.attrs["Description"]["Hostname"]))
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
        return

    def start(self):

        print "\n--- Starting swarm manager on this computer ---\n"
        r = self.engine.swarm.init(advertise_addr=self.swarm_host_ip, listen_addr='0.0.0.0:5000',
                                   force_new_cluster=False, snapshot_interval=5000,
                                   log_entries_for_slow_followers=1200)
        self.key = self.engine.swarm.attrs["JoinTokens"]["Worker"]
        self.__write_key(self.swarm_host)

    def stop(self):

        print "\n--- Stoping the swarm on this computer ---\n"
        machines_on = [str(m.attrs["Description"]["Hostname"]) for m in self.engine.nodes.list()]

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


class Workers(object):

    def __init__(self, engine):

        self.engine = engine

        self.srv_img = IMAGE
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
                self.nodes_name.append(str(m.attrs["Description"]["Hostname"]))

        self.srv_names = [str(s.name) for s in self.engine.services.list()]
        self.srv_ids = [str(s.id) for s in self.engine.services.list()]

        self.workers  = []

    def browse(self):

        self.workers  = []

        for i, srv in enumerate(self.srv_ids):
            try:
                s = self.engine.services.get(srv)
                t = None
                for t2 in s.tasks():
                    if t2["Spec"]["ContainerSpec"]["Image"] == self.srv_img:
                        if t is None:
                            t = t2
                        elif dateutil.parser.parse(t2["Status"]["Timestamp"]) > dateutil.parser.parse(t["Status"]["Timestamp"]):
                            t = t2
                if t is not None:
                    try:
                        host = self.engine.nodes.get(t["NodeID"]).attrs["Description"]["Hostname"]
                        status = t["Status"]["State"]
                        service_name = s.name
                        version = s.version
                        cmd = ""
                        if 'Command' in t["Spec"]["ContainerSpec"].keys():
                            cmd += " ".join(t["Spec"]["ContainerSpec"]["Command"])
                        if 'Args' in t["Spec"]["ContainerSpec"].keys():
                            cmd += " ".join(t["Spec"]["ContainerSpec"]["Args"])
                        if cmd == self.srv_idle_task:
                            cmd_type = "idle"
                        else:
                            cmd_type = "run"

                        # print cmd, status, service_name
                        self.workers.append({"type": cmd_type, "service_version": version,
                                             "service_id": srv, "service_name": service_name,
                                             "host": host, "cmd": cmd, "status": status})

                    except (ValueError, docker.errors.NotFound, KeyError):
                        pass
            except docker.errors.NotFound:
                pass

    def change(self):

        self.browse()

        # Print summary
        wrk_idle = [w for w in self.workers if w["type"] == "idle" ]
        wrk_run = [w for w in self.workers 
                   if (w["type"] == "run" and w["status"] in ["ready", "starting", "running"])]
        wrk_finished =  [w for w in self.workers
                         if (w["type"] == "run" and w["status"] in ["complete", "shutdown"])]

        print "\n--- IDLE Workers: " + str(len(wrk_idle)) + " ---"
        for i, n in enumerate(self.nodes_name):
            print str(n) + ": " + str(len([w for w in wrk_idle if w["host"] == n]))
        print "\n--- RUNNING Workers: " + str(len(wrk_run)) + " ---"
        for i, n in enumerate(self.nodes_name):
            print str(n) + ": " + str(len([w for w in wrk_run if w["host"] == n]))
        print "\n--- FINISHED Workers: " + str(len(wrk_finished)) + " ---"
        for i, n in enumerate(self.nodes_name):
            print str(n) + ": " + str(len([w for w in wrk_finished if w["host"] == n]))


        try:
            to_add = input("\nHow many workers do you want to add (use any negative number to remove): ")
        except (ValueError, TypeError, NameError, SyntaxError):
            print "\nYou should enter a number!"
            return
        if type(to_add) != int:
            print "\nYou should enter a number!"
            return

        if to_add > 0:
            self.add(to_add, len(self.workers))
        elif to_add < 0:
            self.rm()

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
            srv_id = self.engine.services.create(image=self.srv_img, command=self.srv_idle_task, 
                                                 name=self.def_name + name, mode=srv_replicas_mode,
                                                 restart_policy=srv_restart_policy)
        return


class Tasks(object):

    def __init__(self, engine):

        self.engine = engine

        self.cluster = Workers(self.engine)

        self.srv_status = None
        self.running = []

    def process(self, config_list):

        while len(config_list) != 0:

            self.cluster = Workers(self.engine)
            self.cluster.browse()

            n_idle_jobs = len([w for w in self.cluster.workers 
                               if (w["type"] == "idle" and w["status"] == "running")])
            n_desired_jobs = len(config_list)

            if n_idle_jobs > 0:
              print "\n--- Running " + str(min(n_desired_jobs, n_idle_jobs)) + " new jobs. " + \
                    "Still " + str(max(0, n_desired_jobs - n_idle_jobs)) + " in the queue ---\n"

              for _ in range(min(n_desired_jobs, n_idle_jobs)):
                    self.__start_job(config_list.pop(0))

            self.__monitor()

        return []

    def logs(self):

        self.cluster = Workers(self.engine)
        self.cluster.browse()

        worker_idle_name = [w["service_name"] for w in self.cluster.workers if w["type"] == "idle"]
        worker_idle_id = [w["service_id"] for w in self.cluster.workers if w["type"] == "idle"]
        indexes = Picker(title="Select the service to display the logs", 
                         options=worker_idle["service_name"]).getIndex()
        if indexes == []:
            print "No selection. Aborded!"
        else:
            for i in indexes:
                print "\n--- Printing logs from service \"" + str(worker_idle_name[i]) + "\" ---\n"
                ids = self.engine.services.get(worker_idle_id[i])
                logs = ids.logs(details=False, stdout=True, stderr=True, timestamps=False)
                for l in logs:
                    sys.stdout.write(l)

    def __monitor(self):

        if self.srv_status is not None:

            # Check for new or deleted services
            curr = set([w["service_id"] for w in self.cluster.workers])
            old = set([w["service_id"] for w in self.srv_status])
            new_srv = list(curr - old) 
            rm_srv = list(old - curr)
            for s in new_srv:
                self.__add_callback(s)
            for s in rm_srv:
                self.__del_callback(s)

            # Check if version changed
            for old in self.srv_status:
                for new in current:
                    if new["ids"] == old["ids"]:
                        if new["version"] != old["version"]:
                            try:
                                status = self.engine.services.get(new["ids"]).attrs["UpdateStatus"]["State"]
                            except docker.errors.NotFound:
                                break
                            # We can want to store the first version here
                            if status == "completed":
                                self.__modif_callback(new["ids"], old["version"], new["version"])
                        break

        self.srv_status = self.cluster.workers

    def __start_job(self, config):

        # print "--- A new job is started with config: " + str(config) + " ---\n"

        # Pick the first idle service available in the list
        self.cluster.browse()
        srv_ids = [w["service_id"] for w in self.cluster.workers 
                   if (w["type"] == "idle" and w["status"] == "running")][0]
        # print [(w["service_id"], w["type"], w["status"]) for w in self.cluster.workers]
        # print srv_ids
        try:
            srv = self.engine.services.get(srv_ids)
        except docker.errors.NotFound:
            return
        srv_name = srv.name
        self.running.append({"service_name": srv_name, "config": config, "time_init": time.time()})

        # Update the service with the new command
        s2 = srv.update(image=self.cluster.srv_img, command=self.cluster.srv_run_task)
        print str(srv_ids) + ": " + str(s2)

    def __del_callback(self, ids):

        print "\n--- The service " + str(ids) + " has been deleted ---"
        # check results

    def __add_callback(self, ids):

        print "\n--- New service " + str(ids) + " has been added ---"

    def __modif_callback(self, ids, old_v, new_v):

        print "\n--- Service " + str(ids) + " has change from version " + str(old_v) + \
              " to version " + str(new_v) + " ---"
        # Check results


class Manager(object):

    def __init__(self):

        self.engine = docker.from_env()

    def process(self, arg_list):

        if arg_list[0] in ["lan", "computer", "computers", "swarm"]:
            self.process_lan()

        elif arg_list[0] in ["srv", "wrk", "worker", "workers", "service", "services"]:
            self.process_wrk(arg_list)

        elif arg_list[0] in ["tasks", "task", "tsk"]:
            self.process_tsk(arg_list[1:])

        else:
            print "\n--- Invalid argument. Please check the file ---\n"

    def process_lan(self):

        lan = Lan(self.engine)

        if lan.is_init():
            lan.change()
        else:
            lan.start()

    def process_wrk(self, arg_list):

        wrk = Workers(self.engine)

        if len(arg_list) == 1:
            wrk.change()

        else:
            if arg_list[1] in ["add"]:
                if len(arg_list) <= 2:
                    print "\n--- Provide the number of services to add ---\n"
                else:
                    wrk.add(arg_list[2])

            elif arg_list[1] in ["rm", "remove"]:
                wrk.rm()

            elif arg_list[1] in ["change"]:
                wrk.change()

            else:
                print "\n--- Invalid WRK argument. Please check the file ---\n"

    def process_tsk(self, arg_list):

        tsk = Tasks(self.engine)

        if arg_list[0] in ["test", "job", "jobs"]:
            if len(arg_list) > 1:
                config_list = ["default" for i in range(int(arg_list[1]))]
            else:
                config_list = ["default"]
            results = tsk.process(config_list)
            print results

        elif arg_list[0] in ["stop", "leave"]:
            tsk.stop()

        elif arg_list[0] in ["logs", "log"]:
            tsk.logs()

        else:
            print "\n--- Invalid TSK argument. Please check the file ---\n"


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        print "\n--- This manager requires at least one argument (see file) ---\n"
        exit()

    m = Manager()
    m.process(sys.argv[1:])