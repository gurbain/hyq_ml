import docker
import time

dck_cmd = "/bin/bash -c 'cd hyq_ml/hyq; roscore & python physics.py rcf; exit 0'"
dck_img = "hyq:latest"

engine = docker.from_env()
cont = engine.containers.run(dck_img, dck_cmd, detach=True)

logs = cont.logs()

for line in cont.logs(stream=True):
    print (line.strip())