FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04 as ubuntu_hyq

#------------------------
# General setup
#------------------------

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y  --no-install-recommends \
    lsb-release build-essential \
    git sudo dialog apt-utils \
    wget software-properties-common \
    net-tools iputils-ping wget python-pip \
    python-setuptools pyqt5-dev-tools \
    libbluetooth-dev libspnav-dev \
    pyqt4-dev-tools libcwiid-dev \
    cmake gcc g++ qt4-qmake libqt4-dev \
    libusb-dev libftdi-dev \
    python3-defusedxml python-skimage \
    nmap nano htop ncdu tmux

USER root
RUN useradd -ms /bin/bash gurbain
ADD /resources/etc/sudoers.txt /etc/sudoers
RUN chmod 440 /etc/sudoers
USER gurbain
WORKDIR /home/gurbain

#------------------------
# Python Libraries
#------------------------
RUN sudo -H pip install --upgrade travis pip \
         setuptools wheel virtualenv
RUN sudo -H pip install keras tensorflow-gpu \
         tensorflow sklearn tqdm pexpect docker \
         statsmodels mdp gym h5py keras-rl

#------------------------
# Install Hyq_ml repository
#------------------------
RUN git clone https://github.com/gurbain/hyq_ml && \
    cd hyq_ml && git checkout -B gym-rl && git pull origin gym-rl 

#------------------------
# ROS Install
#------------------------
RUN cd hyq_ml/docker/resources/install && \
    bash install_ros.sh

# #------------------------
# # DLS HyQ Install
# #------------------------
RUN cd hyq_ml/docker/resources/install && \
    bash install_dls.sh
RUN sudo chmod a+x hyq_ml/docker/resources/bin/hyq
RUN cd hyq_ml/docker/resources/install && \
    bash setup_shell.sh
RUN cd hyq_ml && sudo python setup.py develop


# #------------------------
# # HyQ Install Hack and Entrypoint
# #------------------------
RUN cd hyq_ml/docker/resources/install && git pull origin master
RUN cd hyq_ml/docker/resources/install && \
    bash hack_me.sh
RUN sudo bash -c 'rm /opt/ros/dls-distro/share/rcf_controller/config/hyq_sim_options.ini \
                  && ln -s /home/gurbain/hyq_ml/config/hyq_sim_options.ini.current_sim \
                  /opt/ros/dls-distro/share/rcf_controller/config/hyq_sim_options.ini'
COPY /resources/bin/entrypoint.sh /usr/local/bin/
RUN sudo chmod a+x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]