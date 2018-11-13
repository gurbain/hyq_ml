#!/bin/bash

source /opt/ros/kinetic/setup.bash
source /opt/ros/dls-distro/setup.bash
sudo apt-get install -y coreutils

timeout 2m /opt/ros/kinetic/bin/roslaunch \
           dls_supervisor operator.launch \
           gazebo:=true osc:=false gui:=false rviz:=false
RETVAL=$?

echo "The Hack finished with the value= $RETVAL"
exit 0