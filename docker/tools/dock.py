import docker
from picker import *
import time
import sys
import subprocess


dck_cmd = "/bin/bash -c 'source /opt/ros/kinetic/setup.bash; \
           source /opt/ros/dls-distro/setup.bash; \
           export PATH=\"/home/gurbain/hyq_ml/docker/bin:$PATH\"; \
           cd /home/gurbain/hyq_ml/hyq; \
           roscore & python physics.py rcf; exit 0'"
dck_img = "hyq:latest"

dck_swarm_user = "gurbain"
dck_swarm_machines = ["paard", "hond", "geit", "kat", "koe", "schaap"]
dck_swarm_key_file = "/home/gurbain/.docker_swarm_key"


def write_swarm_key(key):

    for m in dck_swarm_machines:
        cmd = ['ssh', dck_swarm_user + '@' + m, 'cat - > ' + dck_swarm_key_file]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        p.stdin.write(key)

def read_swarm_key():

    with open(dck_swarm_key_file, 'r') as file:
        key = file.read()
    return key

engine = docker.from_env()

if __name__ == '__main__':

    if len(sys.argv) > 1:
        if sys.argv[1] == "init" or sys.argv[1] == "start":
            r = engine.swarm.init(advertise_addr='172.18.20.20:3030', listen_addr='0.0.0.0:5000',
                            force_new_cluster=False, snapshot_interval=5000,
                            log_entries_for_slow_followers=1200)
            write_swarm_key(engine.swarm.attrs["JoinTokens"]["Worker"])

        if sys.argv[1] == "join":
            engine.swarm.join(remote_addrs=['172.18.20.20:5000'], listen_addr='0.0.0.0:5000',
                            join_token=read_swarm_key())

        if sys.argv[1] == "stop" or sys.argv[1] == "leave":
            engine.swarm.leave(force=True)

        if sys.argv[1] == "ls":

            if len(sys.argv) > 2:
                if sys.argv[2] == "services" or sys.argv[2] == "srv" or sys.argv[2] == "service":
                    print engine.services.list()
                else:
                    print engine.nodes.list()
            else:
                print engine.nodes.list()

        if sys.argv[1] == "jobs":

            if len(sys.argv) > 2:
                jobs_num = int(sys.argv[2])
            else:
                jobs_num = 3

            # Start jobs_num jobs
            for i in range(jobs_num):
                srv_replicas_mode = docker.types.ServiceMode("replicated", replicas=1)
                srv_restart_policy =  docker.types.RestartPolicy(condition='none', 
                                                           delay=0, max_attempts=0, 
                                                           window=0) 
                srv_id = engine.services.create(image=dck_img, command=dck_cmd, name="sim_job_" + str(i),
                    restart_policy=srv_restart_policy, mode=srv_replicas_mode)

        if sys.argv[1] == "logs":

            # Check the running services
            srv_names = [str(s.name) for s in engine.services.list()]
            srv_ids = [str(s.id) for s in engine.services.list()]
            indexes = Picker(title="Select the service to display the logs", options=srv_names).getIndex()
            if indexes == []:
                print "No selection. Aborded!"
            else:
                for i in indexes:
                    print "\n--- Printing logs from service \"" + str(srv_names[i]) + "\" ---\n"
                    ids = engine.services.get(srv_ids[i])
                    logs = ids.logs(details=False, stdout=True, stderr=True, timestamps=False)
                    for l in logs:
                        sys.stdout.write(l)

        if sys.argv[1] == "rm" or  sys.argv[1] == "remove":

            # Check the running services
            srv_names = [str(s.name) for s in engine.services.list()]
            srv_ids = [str(s.id) for s in engine.services.list()]
            indexes = Picker(title="Select the services to kill and remove", options=srv_names).getIndex()
            if indexes == []:
                print "No selection. Aborded!"
            else:
                for i in indexes:
                    print "\n--- Killing and deleting service \"" + str(srv_names[i]) + "\" ---\n"
                    ids = engine.services.get(srv_ids[i])
                    ids.remove()

        if sys.argv[1] == "tasks":

            # Check the running services
            srv_names = [str(s.name) for s in engine.services.list()]
            srv_ids = [str(s.id) for s in engine.services.list()]
            indexes = Picker(title="Select the services for which the task shall be displayed", 
                             options=srv_names).getIndex()
            if indexes == []:
                print "No selection. Aborded!"
            else:
                for i in indexes:
                    print "\n--- List of tasks from service \"" + str(srv_names[i]) + "\" ---\n"
                    ids = engine.services.get(srv_ids[i])
                    print ids.tasks()