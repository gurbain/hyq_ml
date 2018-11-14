from concurrent.futures import ThreadPoolExecutor
import docker
import time
import sys


dck_cmd = "/bin/bash -c 'cd hyq_ml/hyq; roscore & python physics.py rcf; exit 0'"
dck_img = "hyq:latest"
dck_max_jobs = 50


# A thred that manage a simulation run
def threaded_job(client):

    cont = dock.containers.run(dck_img, dck_cmd, detach=True)
    for line in cont.logs(stream=True):
        print(line.strip())


dock = docker.from_env()

if __name__ == '__main__':

    if len(sys.argv) > 1:
        if sys.argv[1] == "init":
            dock.swarm.init(advertise_addr='eth0', listen_addr='0.0.0.0:5000',
                            force_new_cluster=False, snapshot_interval=5000,
                            log_entries_for_slow_followers=1200)

        if sys.argv[1] == "stop" or sys.argv[1] == "leave":
            dock.swarm.leave(force=True)


        if sys.argv[1] == "test":

            # Create a threaded pool that manage the jobs
            if len(sys.argv) > 2:
                threads_num = sys.argv[2]
            else:
                threads_num = 10
            client = docker.DockerClient(base_url='tcp://localhost:5000',  tls=False)
            pool = ThreadPoolExecutor(dck_max_jobs)

            for my_message in range(threads_num):
                future = pool.submit(threaded_job, client)