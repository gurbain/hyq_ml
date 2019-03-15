#!/bin/bash

docker run -it --rm  gym_hyq:latest /bin/bash -c 'source /opt/ros/kinetic/setup.bash && \
                                                  source /opt/ros/dls-distro/setup.bash && \
                                                  roslaunch dls_supervisor operator.launch gazebo:=true osc:=false gui:=false rviz:=false'


