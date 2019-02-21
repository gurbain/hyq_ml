#!/bin/bash

# HyQ
sudo apt-get install -y git python-dateutil python-pip 
sudo pip install docker numpy
git clone http://github.com/gurbain/hyq_ml
cd hyq_ml/docker/ && ./build.sh

# NFS Client
sudo ln -s /users/gurbain /home/gurbain && chown gurbain:wal-humanbrainpr /home/gurbain 
sudo bash -c 'echo "paard:/export/gurbain   /users/gurbain/docker_sim    nfs   rw,exec,auto,user    0    0" >> /etc/fstab'
mkdir -p "/users/gurbain/docker_sim"
sudo mount -a

# Usefull tools
sudo apt-get install -y nmap nano htop ncdu tmux


## WARNING!!
## To do and check manually

# Adapt /etc/hosts.deny by commenting the RPC line : sudo nano /etc/hosts.deny 
# Try ssh access from every machine to every machine: ssh paard && ssh kat && ssh geit && ssh schaap
# Check the .zshrc everywhere
# Add deb repositories on schaap
# Remove warning in sudo nano /usr/local/lib/python2.7/dist-packages/requests/__init__.py