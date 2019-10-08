#!/bin/bash

docker run -it --rm  hyq:latest /bin/bash -c 'source /opt/ros/dls-distro/setup.bash; \
                                        cd /home/gurbain/hyq_ml/hyq; \
                                        mkdir /home/gurbain/a; \
                                        ln -s /home/gurbain/hyq_ml/config/sim_default.txt \
                                        /home/gurbain/a/config.txt; \
                                        roscore & python simulation.py /home/gurbain/a'

