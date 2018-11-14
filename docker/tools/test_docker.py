import docker
import time

dck_cmd = "/bin/bash -c 'cd hyq_ml/hyq; roscore & python physics.py rcf'"
dck_img = "hyq:latest"

dock = docker.from_env()
cont = dock.containers.run(dck_img, dck_cmd, detach=True)

for i in range(50):
	print cont.status
	print cont.logs(), cont.status
	time.sleep(1)
cont.stop()