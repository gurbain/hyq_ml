import docker
import time

dck_cmd = "/bin/bash -c 'cd hyq_ml/hyq; roscore & python physics.py rcf; exit 0'"
dck_img = "hyq:latest"

dock = docker.from_env()
cont = dock.containers.run(dck_img, dck_cmd, detach=True)

for i in range(100):
	print cont.status
	time.sleep(5)
cont.stop()