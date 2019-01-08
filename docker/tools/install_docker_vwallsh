#!/bin/bash
# IPV4 and Linux
wget -O - -q https://www.wall2.ilabt.iminds.be/enable-nat.sh | sudo bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    linux-image-extra-$(uname -r) \
    linux-image-extra-virtual
sudo apt-get -y --no-install-recommends install curl apt-transport-https ca-certificates curl software-properties-common
sudo apt-get remove docker docker-engine docker.io

# Set Brussels time
sudo cp /usr/share/zoneinfo/Europe/London /etc/localtime

# Create SSH keys
ssh-keygen -t rsa -b 4096 -C "gabriel.urbain@ugent.be"

# Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo mkdir /mnt/docker
sudo ln -s /mnt/docker /var/lib/docker
sudo apt-get -y install docker-ce
sudo usermod -aG docker $USER

# ZSH
sudo apt-get install -y git zsh
git clone https://github.com/gurbain/customization
cd customization && mv .zshrc .oh-my-zsh ..
cd $HOME
head -n -14 .zshrc >> .zshrc  
cd $HOME && rm -fr customization
sudo bash -c 'passwd gurbain'
chsh -s $(which zsh)
reset

