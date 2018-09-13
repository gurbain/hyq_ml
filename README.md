HyQ Machine Learning Project
============================

This repository contains a series of scripts to:
 - Start, stop, control and monitor the HyQ simulations
 - Interface with the HyQ RCF Controller over ROS
 - Process and visualize the data extracted from the HyQ simulations
 - Implement Echo State Network with RLS (numpy) and Backpropagation (keras) rules
 - Manipulate a network ressource for simulataneous training, running and analysis in a multi-threaded fashion


Installation
------------

Requirements:
 - numpy
 - tensorflow
 - keras
 - ros kinetcic suite (including rosbag, rospy and rospackage)


Usage
-----

Running a HyQ simulation

Learning a predictive model for the simulation data from the controller saved in a bag file

Start the framework HyQ simulation/NN training
