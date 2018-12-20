#!/bin/bash

source ./fun.cfg

# What do you want to install my dear whippersnapper?
if [[ $# -eq 3 ]]; then
    INSTALL_DEPS=$1
    INSTALL_EXTRAS=$2
    RELEASE=$3 #stable or latest
else
    INSTALL_DEPS=true
    INSTALL_EXTRAS=false
    RELEASE= #empty means take the stable pkgs
fi

PKGS_FOLDER="./pkgs_list"

# Add custom dls ppa
dls_ppa_setting

# Add custom dls ros repo
dls_ros_repo_setting

# Silent java questions about licence!
echo "oracle-java8-installer shared/accepted-oracle-license-v1-1 select true" | sudo debconf-set-selections

# Add external ppas
sudo add-apt-repository --yes ppa:webupd8team/java
#sudo add-apt-repository --yes ppa:xqms/opencv-nonfree
sudo add-apt-repository --yes ppa:danielrichter2007/grub-customizer

#Add Eigen backport PPA
export LANG=C.UTF-8
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys ECD154D280FEB8AC
sudo add-apt-repository --yes ppa:nschloe/eigen-backports

# Remove apt cache and update the sources
sudo rm -rf /var/lib/apt/lists/* 
sudo apt-get update

if $INSTALL_DEPS; then
	echo -e "${COLOR_INFO}Install $RELEASE dependencies ${COLOR_RESET}"
	# Install external deps
	cat $PKGS_FOLDER/ext_dependencies_list.txt | grep -v \# | xargs sudo apt-get install -y
	# Install python pkgs with pip:
	cat $PKGS_FOLDER/python_list.txt  | grep -v \# | xargs pip install	
else
	echo -e "${COLOR_INFO}No dependencies have been installed...${COLOR_RESET}"
fi

if $INSTALL_EXTRAS; then
	echo -e "${COLOR_INFO}Install extra programs and tools${COLOR_RESET}"
	# Install extra stuff
	cat $PKGS_FOLDER/extras_list.txt | grep -v \# | xargs sudo apt-get install -y
else
	echo -e "${COLOR_INFO}No extra tools have been installed...${COLOR_RESET}"
fi

# Install DLS Itself
sudo apt-get install -y dls-distro

#Install git lfs
wget https://github.com/git-lfs/git-lfs/releases/download/v2.3.4/git-lfs-linux-amd64-2.3.4.tar.gz -P /tmp/
tar xvf /tmp/git-lfs-linux-amd64-2.3.4.tar.gz -C /tmp/
sudo /tmp/git-lfs-2.3.4/install.sh

# Install gtest
if [ -d "/usr/src/gtest" ]; then
	cd /usr/src/gtest
	sudo cmake CMakeLists.txt
	sudo make
	sudo cp *.a /usr/lib
fi

