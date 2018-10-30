HyQ Machine Learning Project
============================

This repository contains a series of scripts to:
 - Start, stop, control and monitor the HyQ simulations
 - Interface with the HyQ RCF Controller over ROS
 - Process and visualize the data extracted from the HyQ simulations
 - Implement Echo State Network with RLS (numpy) and Backpropagation (keras) rules
 - Manipulate a network ressource for simulataneous training, running and analysis in a multi-threaded fashion


Classical Installation
----------------------

This project needs the ROS DLS framework packages. Once installed, you can simply clone the repository:
```bash
git clone https://github.com/gurbain/hyq_ml
```
To work with an example, fist check-out the last version and install the libraries in dev mode:
```bash
cd hyq_ml && git checkout embodiment && sudo python setup.py develop
```
Or in install mode (not advised as local changes are not taken into account):
```bash
cd hyq_ml && git checkout embodiment && sudo python setup.py install --record installed_files.txt
```
You can then remove the library with:
```bash
cd hyq_ml && cat installed_files.txt | sudo xargs rm -rf
```


Conda Installation
------------------

A list of requirements can be found in the file *requirements.txt*. However, to install a clean functional environment, you can also use conda:
```bash
cd hyq_ml && conda env create -f hyq.yml
source activate hyq
```


Docker Installation
-------------------

An alternative on the IDLab local network, is to install both the dependencies and the DLS packages from a local debian package server using docker. The two previous section are now replpaced with:
```bash
git clone https://github.com/gurbain/hyq_ml
cd hyq_ml/docker && bash build.sh && bash run.sh
```
This is particularly useful when running on a remote server. A set of tools is available in the folder *docker/tools* to monitor and view how the simulation is behaving in the background.


Usage
-----

This section is not yet documented! 

== Running a HyQ simulation ==

```bash
source activate hyq
cd hyq_ml &&  pip install .
cd experiments  && python evaluate_kp_kd_space.py    
```

== Learning a predictive model for the simulation data from the controller saved in a bag file ==

== Start the framework HyQ simulation/NN training ==
