#!/bin/bash
echo "Installing ROS..."
if [[ $# > 0 ]]; then
    ROS_DISTRO=$1
else
    ROS_DISTRO=kinetic
fi

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros-latest.list'
wget https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get -y install ros-$ROS_DISTRO-desktop-full
sudo apt-get -y install ros-$ROS_DISTRO-roslint
sudo apt-get -y install ros-$ROS_DISTRO-rqt
sudo rosdep init
rosdep update

echo "Installing ROS-Control..."
sudo apt-get -y install ros-${ROS_DISTRO}-ros-control ros-${ROS_DISTRO}-ros-controllers ros-${ROS_DISTRO}-realtime-tools ros-${ROS_DISTRO}-gazebo-ros-control
echo "Installing ROS multisense..."
sudo apt-get -y install ros-${ROS_DISTRO}-multisense ros-${ROS_DISTRO}-multisense-description ros-${ROS_DISTRO}-multisense-lib
echo "Installing ROS grid-map..."
sudo apt-get -y install ros-${ROS_DISTRO}-grid-map ros-${ROS_DISTRO}-grid-map-pcl
echo "Installing ROS useful stuff and goodies..."
sudo apt-get -y install ros-${ROS_DISTRO}-gazebo-ros-pkgs ros-${ROS_DISTRO}-ps3joy ros-${ROS_DISTRO}-joy ros-${ROS_DISTRO}-octomap ros-${ROS_DISTRO}-stereo-image-proc ros-${ROS_DISTRO}-opencv3 ros-${ROS_DISTRO}-robot-state-publisher ros-${ROS_DISTRO}-openni2-camera

# Setup Bashrc
#if grep -Fwq ${ROS_DISTRO} ~/.bashrc
#then 
# 	echo -e "Bashrc already updated, skipping this step..."
#else
#    	echo -e "Update the bashrc."
#	echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
#fi
#
#source ~/.bashrc
