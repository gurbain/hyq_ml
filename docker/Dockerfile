FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04 as ubuntu_hyq

#------------------------
# General setup
#------------------------
RUN apt-get update && \
    apt-get install -y  --no-install-recommends \
    lsb-release build-essential \
    git sudo dialog apt-utils \
    wget software-properties-common \
    net-tools iputils-ping wget python-pip \
    python-setuptools

USER root
RUN useradd -ms /bin/bash gurbain
ADD /etc/sudoers.txt /etc/sudoers
RUN chmod 440 /etc/sudoers
USER gurbain
WORKDIR /home/gurbain

#------------------------
# Dependencies
#------------------------
RUN sudo pip install keras tensorflow-gpu \
         tensorflow sklearn tqdm pexpect

#------------------------
# Install Hyq_ml repository
#------------------------
RUN pwd
RUN git clone https://github.com/gurbain/hyq_ml

#------------------------
# ROS Install
#------------------------
RUN cd hyq_ml/docker/install && \
    bash install_ros.sh

# #------------------------
# # DLS HyQ Install
# #------------------------
RUN cd hyq_ml/docker/install && \
    bash install_dls.sh
RUN sudo chmod a+x hyq_ml/docker/bin/hyq
RUN cd hyq_ml/docker/install && \
    bash setup_shell.sh
RUN cd hyq_ml && sudo python setup.py develop


# #------------------------
# # HyQ Install Hack
# #------------------------
RUN cd hyq_ml/docker/install && \
    bash hack_me.sh



