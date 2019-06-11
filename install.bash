
# APT Dependencies
apt-get update && \
    apt-get install -y  --no-install-recommends \
    lsb-release build-essential \
    git sudo dialog apt-utils \
    wget software-properties-common \
    net-tools iputils-ping wget python-pip \
    python-setuptools
apt-get install -y nmap nano htop ncdu tmux

# PIP Dependencies
sudo -H pip install --upgrade travis pip \
    setuptools wheel virtualenv
sudo -H pip install keras tensorflow-gpu \
    tensorflow sklearn tqdm pexpect docker \
    statsmodels mdp

# GIT Dependencies
git clone https://github.com/gurbain/entropy
sudo bash -c 'cd entropy && pip install -r requirements.txt && \
              python setup.py develop'


# HyQ ML
git clone https://github.com/gurbain/hyq_ml
cd hyq_ml && sudo python setup.py develop
