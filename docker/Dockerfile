FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

#------------------------
# General setup
#------------------------
RUN apt-get update && \
    apt-get install -y lsb-release \
    git sudo dialog apt-utils\
    wget software-properties-common \
    net-tools iputils-ping

USER root
RUN useradd -ms /bin/bash gurbain
ADD sudoers.txt /etc/sudoers
RUN chmod 440 /etc/sudoers
USER gurbain
WORKDIR /home/gurbain

#------------------------
# Install Hyq_ml repository
#------------------------
RUN git clone https://github.com/gurbain/hyq_ml
RUN cd hyq_ml && \
    git checkout embodiment && \
    git pull origin embodiment

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
RUN cd hyq_ml/docker/install_hyq && \
    bash setup_shell.sh