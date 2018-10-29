#!/bin/zsh

# ROS Version
ROS_DISTRO=kinetic

# Get current shell
CURR_SHELL=$(echo $SHELL | rev | cut -d/ -f1 | rev)
if [ -z "$CURR_SHELL" ]
then
      CURR_SHELL='sh'
fi
echo "Current Shell: ${CURR_SHELL}"

# Configure shell
if grep -Fwq ${ROS_DISTRO} ~/."${CURR_SHELL}rc"
then 
 	echo "${CURR_SHELL}rc already updated, skipping this step..."
else
    	echo "Update the ${CURR_SHELL}rc"
	echo "## Source ROS and HyQ Packages" >> ~/."${CURR_SHELL}rc"
	echo "source /opt/ros/${ROS_DISTRO}/setup.${CURR_SHELL}" >> ~/."${CURR_SHELL}rc"
	echo "source /opt/ros/dls-distro/setup.${CURR_SHELL}" >> ~/."${CURR_SHELL}rc"

fi

source ~/."${CURR_SHELL}rc"
