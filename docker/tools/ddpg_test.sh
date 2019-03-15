#!/bin/bash

DOCKER_IMG="gym_hyq"
DOCKER_TAG="latest"

docker run -it --rm  $DOCKER_IMG:$DOCKER_TAG  /bin/bash -c 'source /opt/ros/dls-distro/setup.bash; \
                                                          cd /home/gurbain/hyq_ml/agents/ddpg; \
                                                          python train.py'
