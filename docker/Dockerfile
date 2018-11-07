FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04 as ubuntu_hyq

#------------------------
# General setup
#------------------------
RUN apt-get update && \
    apt-get install -y  --no-install-recommends \
    lsb-release build-essential \
    git sudo dialog apt-utils \
    wget software-properties-common \
    net-tools iputils-ping wget

USER root
RUN useradd -ms /bin/bash gurbain
ADD /etc/sudoers.txt /etc/sudoers
RUN chmod 440 /etc/sudoers
USER gurbain
WORKDIR /home/gurbain

#------------------------
# Install Hyq_ml repository
#------------------------
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


## Test
RUN apt-get update &&\
    apt-get install -y avahi-utils avahi-daemon libnss-mdns systemd
RUN cd /lib/systemd/system/sysinit.target.wants/;\
  ls | grep -v systemd-tmpfiles-setup | xargs rm -f $1 \
    rm -f /lib/systemd/system/multi-user.target.wants/*;\
  rm -f /etc/systemd/system/*.wants/*;\
  rm -f /lib/systemd/system/local-fs.target.wants/*; \
  rm -f /lib/systemd/system/sockets.target.wants/*udev*; \
  rm -f /lib/systemd/system/sockets.target.wants/*initctl*; \
  rm -f /lib/systemd/system/basic.target.wants/*;\
  rm -f /lib/systemd/system/anaconda.target.wants/*; \
  rm -f /lib/systemd/system/plymouth*; \
  rm -f /lib/systemd/system/systemd-update-utmp*
RUN mkdir -p /var/run/dbus
ENV init /lib/systemd/systemd

## Why??
#RUN cd hyq_ml/docker/install && \
#    bash hack_me.sh