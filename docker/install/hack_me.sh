#!/bin/bash

source /opt/ros/kinetic/setup.bash
source /opt/ros/dls-distro/setup.bash


sudo apt-get install -y coreutils

timeout 25 /opt/ros/kinetic/bin/roslaunch \
           dls_supervisor operator.launch \
           gazebo:=true osc:=false gui:=false rviz:=false