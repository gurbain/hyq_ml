#!/bin/bash

DOCKER_IMG="gym_hyq"
DOCKER_TAG="latest"

DATA_FOLDER="/home/gurbain/docker_sim"

docker run -v  $DATA_FOLDER:$DATA_FOLDER \
           -it --rm  $DOCKER_IMG:$DOCKER_TAG  \
           /bin/bash -c 'source /opt/ros/dls-distro/setup.bash; \
                         cd /home/gurbain/hyq_ml/agents/ddpg; \
                         python train.py'
