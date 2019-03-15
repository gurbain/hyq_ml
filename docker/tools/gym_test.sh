#!/bin/bash

docker run -it --rm  gurbain/gym_hyq:latest /bin/bash -c 'source /opt/ros/dls-distro/setup.bash; \
                                                          cd /home/gurbain/hyq_ml/gym_hyq/; \
                                                          python hyq_env_test.py'