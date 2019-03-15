#!/bin/bash

docker run -it --rm  gurbain/gym_hyq:latest /bin/bash -c 'source /opt/ros/dls-distro/setup.bash; \
                                                          cd /home/gurbain/hyq_ml/agents/ddpg; \
                                                          python train.py'
