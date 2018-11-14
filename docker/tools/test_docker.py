import docker
import time

dck_cmd = "cd hyq_ml/experiments; roscore & python evaluate_kp_kd_space.py"
dck_img = "hyq:latest"

dock = docker.from_env()
cont = dock.containers.run(dck_img, dck_cmd, auto_remove=True)#, detach=True)
print cont
time.sleep(10)
print cont.logs()
# time.sleep(2)
# print cont.logs()
# cont.stop()