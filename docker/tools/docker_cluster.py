from concurrent.futures import ThreadPoolExecutor
import docker
import time
import sys
import subprocess


dck_cmd = "/bin/bash -c 'cd hyq_ml/hyq; roscore & python physics.py rcf; exit 0'"
dck_img = "hyq:latest"
dck_max_jobs = 50

dck_swarm_user = "gurbain"
dck_swarm_machines = ["paard", "hond", "geit"]
dck_swarm_key_file = "/home/gurbain/.docker_swarm_key"


# A thred that manage a simulation run
def threaded_job(client):

    cont = dock.containers.run(dck_img, dck_cmd, detach=True)
    for line in cont.logs(stream=True):
        print(line.strip())

def write_swarm_key(key):

    for m in dck_swarm_machines:
        cmd = ['ssh', dck_swarm_user + '@' + m, 'cat - > ' + dck_swarm_key_file]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        p.stdin.write(key)

def read_swarm_key():

    with open(dck_swarm_key_file, 'r') as file:
        key = file.read()
    return key

dock = docker.from_env()

if __name__ == '__main__':

    if len(sys.argv) > 1:
        if sys.argv[1] == "init":
            r = dock.swarm.init(advertise_addr='172.18.20.20', listen_addr='0.0.0.0:5000',
                            force_new_cluster=False, snapshot_interval=5000,
                            log_entries_for_slow_followers=1200)
            write_swarm_key(dock.swarm.attrs["JoinTokens"]["Worker"])

        if sys.argv[1] == "join":
            dock.swarm.join(remote_addrs=['172.18.20.20:5000'], join_token=read_swarm_key())

        if sys.argv[1] == "stop" or sys.argv[1] == "leave":
            dock.swarm.leave(force=True)

        if sys.argv[1] == "test":

            # Create a threaded pool that manage the jobs
            if len(sys.argv) > 2:
                threads_num = sys.argv[2]
            else:
                threads_num = 10
            client = docker.from_env()
            print client
            pool = ThreadPoolExecutor(dck_max_jobs)

            for my_message in range(threads_num):
                future = pool.submit(threaded_job, client)