# SKlearn forces warnings...
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from collections import deque
import ConfigParser
import copy
import datetime
from keras import backend as K
import math
#  import matplotlib.pyplot as plt
import numpy as np
import pause
import pexpect
import pickle
import psutil
import rospy as ros
import rosbag
from scipy.interpolate import interp1d
import sys
import tensorflow as tf
import threading
import time

from std_msgs.msg import Bool, Float64, Float64MultiArray, MultiArrayLayout, Header
from std_srvs.srv import Trigger, TriggerResponse

import control
import physics
import network
import utils


class Simulation(object):

    def __init__(self, folder=None):

        # Class variable
        self.config = dict()
        self.t_sim = 0
        self.t_train = 0
        self.t_start_cl = 0
        self.t_stop_cl = 0
        self.t_cl = 0
        self.plot = None
        self.verbose = None
        self.publish_actions = None
        self.publish_states = None
        self.publish_loss = None
        self.publish_error = None
        self.sim_file = None
        self.ol = None
        self.epoch_num = 0
        self.train_buff_size = 0
        self.sm = None
        self.train = None
        self.play_from_file = False
        self.time_step = 0

        # Physics variable
        self.view = False
        self.init_impedance = None
        self.remote = False
        self.real_time = False

        # Saving
        self.save_ctrl = False
        self.save_ctrl_state = []
        self.save_ctrl_action = []
        self.save_states = False
        self.save_states_t_real = []
        self.save_states_x = []
        self.save_states_y = []
        self.save_states_z = []
        self.save_states_phi = []
        self.save_states_theta = []
        self.save_states_psi = []
        self.save_states_t_trot = 0
        self.save_metrics = False
        self.save_stop_train_i = 0
        self.save_start_test_i = 0
        self.save_index = 0

        # Class uninitialized objects
        self.physics = None
        self.network = None
        self.target_fct = None
        self.state_1_fct = None
        self.state_2_fct = None

        # Execution variables
        self.t = 0
        self.it = 0
        self.last_action_it = 0
        self.last_state_it = 0
        self.train_it = 0
        self.nn_started = False
        self.train_session = None
        self.pred_session = None
        self.train_graph = None
        self.pred_graph = None
        self.train_pred_lock = threading.Lock()

        # Training variable
        self.t_hist = []
        self.state = deque([])
        self.x_train_step = []
        self.y_train_step = []
        self.loss = 1.0
        self.accuracy = 0
        self.nn_weight = 0
        self.training_thread = None
        self.train_sm_mode = "Training"

        # Plot tools
        self.plot_ps = None
        self.ps_to_kill = "plotjuggler"

        # ROS variables
        self.curr_state_pub_name = "/sim_debug/curr_states"
        self.rec_state_pub_name = "/sim_debug/rec_states"
        self.pred_pub_name = "/sim_debug/prediction"
        self.target_pub_name = "/sim_debug/target"
        self.mix_pub_name = "/sim_debug/mix_cl"
        self.loss_pub_name = "/sim_debug/prediction_loss"
        self.acc_pub_name = "/sim_debug/prediction_acc"
        self.sm_ser_name = "/sim/rcf_nn_transition"
        self.pub_curr_state = None
        self.pub_rec_state = None
        self.pub_pred = None
        self.pub_target = None
        self.pub_mix = None
        self.pub_loss = None
        self.pub_acc = None
        self.ser_sm = None

        # Parse the simulation folder 
        self.folder = folder
        self.__parse_folder()

    def __parse_folder(self):

        # Open the config file and retrieve the data
        config_parser = ConfigParser.ConfigParser()
        config_parser.read(self.folder + "/config.txt")
        self.config = {s: dict(config_parser.items(s)) for s in config_parser.sections()}

        self.t_sim = float(self.config["Timing"]["t_sim"])
        self.t_train = float(self.config["Timing"]["t_train"])
        self.t_start_cl = float(self.config["Timing"]["t_start_cl"])
        self.t_stop_cl = float(self.config["Timing"]["t_stop_cl"])
        self.t_cl = self.t_stop_cl - self.t_start_cl

        # Class variable
        self.plot = eval(self.config["Debug"]["plot"])
        self.verbose = int(self.config["Debug"]["verbose"])
        self.view = eval(self.config["Debug"]["view"])

        self.sm = eval(self.config["Simulation"]["sm"])
        self.ol = eval(self.config["Simulation"]["ol"])
        self.publish_actions = eval(self.config["Simulation"]["pub_actions"])
        self.publish_states = eval(self.config["Simulation"]["pub_states"])
        self.publish_loss = eval(self.config["Simulation"]["pub_loss"])
        self.publish_error = eval(self.config["Simulation"]["pub_error"])
        self.save_ctrl = eval(self.config["Simulation"]["save_ctrl"])
        self.save_states = eval(self.config["Simulation"]["save_states"])
        self.save_metrics = eval(self.config["Simulation"]["save_metrics"])
        self.time_step = float(self.config["Simulation"]["time_step"])

        self.epoch_num = int(self.config["Training"]["epoch_num"])
        self.train = eval(self.config["Training"]["train"])
        self.train_buff_size = int(self.config["Training"]["train_buff_size"])

        self.init_impedance = eval(self.config["Physics"]["init_impedance"])
        self.remote = eval(self.config["Physics"]["remote_server"])
        self.real_time = eval(self.config["Physics"]["real_time"])

        self.sim_file = self.config["Network"]["sim_file"]

    def start(self):

        # Create and start the handle thread to the physics simulation
        self.physics = physics.HyQSim(init_impedance=self.init_impedance,
                                      view=self.view,
                                      remote=self.remote,
                                      verbose=self.verbose,
                                      rt=self.real_time,
                                      publish_error=self.publish_error,)
        ros.init_node("simulation", anonymous=True)
        self.physics.start()

        # If sim folder is specified create a untrained network and read simulation from a FILE
        if self.sim_file != "None":
            self.play_from_file = True
            self.network = network.NN(data_file=self.sim_file,
                                      save_folder=self.folder,
                                      verbose=int(self.config["Network"]["verbose"]),
                                      nn_layers=eval(self.config["Network"]["nn_struct"]),
                                      test_split=float(self.config["Network"]["test_split"]),
                                      val_split=float(self.config["Network"]["val_split"]),
                                      stop_delta=float(self.config["Network"]["stop_delta"]),
                                      stop_pat=int(self.config["Network"]["stop_pat"]),
                                      optim=self.config["Network"]["optim"],
                                      metric=self.config["Network"]["metric"],
                                      batch_size=int(self.config["Network"]["batch_size"]),
                                      max_epochs=int(self.config["Network"]["max_epochs"]),
                                      regularization=float(self.config["Network"]["regularization"]),
                                      esn_n_res=int(self.config["Network"]["esn_n_res"]),
                                      esn_n_read=int(self.config["Network"]["esn_n_read"]),
                                      esn_in_mask=eval(self.config["Network"]["esn_in_mask"]),
                                      esn_out_mask=eval(self.config["Network"]["esn_out_mask"]),
                                      esn_real_fb=eval(self.config["Network"]["esn_real_fb"]),
                                      esn_spec_rad=float(self.config["Network"]["esn_spec_rad"]),
                                      esn_damping=float(self.config["Network"]["esn_damping"]),
                                      esn_sparsity=float(self.config["Network"]["esn_sparsity"]),
                                      esn_noise=float(self.config["Network"]["esn_noise"]),
                                      checkpoint=eval(self.config["Network"]["checkpoint"]),
                                      no_callbacks=eval(self.config["Network"]["no_callbacks"]),
                                      random_state=int(self.config["Network"]["random_state"]))

            self.network.load_data(True)

        # Else, load the trained network and start a REAL simulation
        # elif self.nn_folder is not None: # this check has to be adatapted
        #     self.play_from_file = False
        #     self.network = network.NN(max_epochs=100000,
        #                               checkpoint=False,
        #                               no_callbacks=True,
        #                               esn_real_fb=True,
        #                               verbose=0,
        #                               nn_layers=eval(self.config["Network"]["nn_struct"]),
        #                               test_split=float(self.config["Network"]["test_split"]),
        #                               val_split=float(self.config["Network"]["val_split"]),
        #                               stop_delta=float(self.config["Network"]["stop_delta"]),
        #                               stop_pat=int(self.config["Network"]["stop_pat"]),
        #                               optim=self.config["Network"]["optim"],
        #                               metric=self.config["Network"]["metric"],
        #                               batch_size=int(self.config["Network"]["batch_size"]),
        #                               regularization=float(self.config["Network"]["regularization"]),
        #                               esn_n_res=int(self.config["Network"]["esn_n_res"]),
        #                               esn_n_read=int(self.config["Network"]["esn_n_read"]),
        #                               esn_in_mask=eval(self.config["Network"]["esn_in_mask"]),
        #                               esn_out_mask=eval(self.config["Network"]["esn_out_mask"]),
        #                               esn_spec_rad=float(self.config["Network"]["esn_spec_rad"]),
        #                               esn_damping=float(self.config["Network"]["esn_damping"]),
        #                               esn_sparsity=float(self.config["Network"]["esn_sparsity"]),
        #                               esn_noise=float(self.config["Network"]["esn_noise"]),
        #                               checkpoint=eval(self.config["Network"]["checkpoint"]),
        #                               random_state=int(self.config["Network"]["random_state"]))
        #     self.network.load(self.nn_folder, load_all=True)
        #     self.network.esn.verbose = 0
        #     self.train_it += 1

        else:
            self.play_from_file = False
            self.network = network.NN(max_epochs=int(self.config["Network"]["max_epochs"]),
                                      checkpoint=False,
                                      no_callbacks=True,
                                      verbose=int(self.config["Network"]["verbose"]),
                                      save_folder=self.folder,
                                      nn_layers=eval(self.config["Network"]["nn_struct"]),
                                      test_split=float(self.config["Network"]["test_split"]),
                                      val_split=float(self.config["Network"]["val_split"]),
                                      stop_delta=float(self.config["Network"]["stop_delta"]),
                                      stop_pat=int(self.config["Network"]["stop_pat"]),
                                      optim=self.config["Network"]["optim"],
                                      metric=self.config["Network"]["metric"],
                                      batch_size=int(self.config["Network"]["batch_size"]),
                                      regularization=float(self.config["Network"]["regularization"]),
                                      esn_n_res=int(self.config["Network"]["esn_n_res"]),
                                      esn_n_read=int(self.config["Network"]["esn_n_read"]),
                                      esn_in_mask=eval(self.config["Network"]["esn_in_mask"]),
                                      esn_out_mask=eval(self.config["Network"]["esn_out_mask"]),
                                      esn_real_fb=eval(self.config["Network"]["esn_real_fb"]),
                                      esn_spec_rad=float(self.config["Network"]["esn_spec_rad"]),
                                      esn_damping=float(self.config["Network"]["esn_damping"]),
                                      esn_sparsity=float(self.config["Network"]["esn_sparsity"]),
                                      esn_noise=float(self.config["Network"]["esn_noise"]),
                                      random_state=int(self.config["Network"]["random_state"]))

        # Retrieve interpolation functions for the target and del the rest
        if self.play_from_file:
            self.target_fct = self.network.get_fct(self.network.y_t,
                                                   self.network.y_val)
            self.state_1_fct = self.network.get_fct(self.network.x_t,
                                                    self.network.x_val)
            if self.network.x2_t.shape[0] is not 0:
                self.state_2_fct = self.network.get_fct(self.network.x2_t,
                                                        self.network.x2_val)

            if max(self.network.y_t) - 1 < self.t_sim:
                self.printv("\nThe nn data recorded file is too short. "
                            "Lowering the simulation time from {:.2f}s".format(self.t_sim) +
                            " to {:.2f}s".format(max(self.network.y_t) - 1))
                self.t_sim = max(self.network.y_t) - 1
            del self.network.y_t, self.network.y_val, \
                self.network.x_t, self.network.x_val, \
                self.network.x2_t, self.network.x2_val

        # Create the ROS topics to debug simulation
        if self.publish_states:
            self.pub_curr_state = ros.Publisher(self.curr_state_pub_name,
                                                Float64MultiArray,
                                                queue_size=1)
            self.pub_rec_state = ros.Publisher(self.rec_state_pub_name,
                                               Float64MultiArray,
                                               queue_size=1)
        if self.publish_actions:
            self.pub_pred = ros.Publisher(self.pred_pub_name,
                                          Float64MultiArray,
                                          queue_size=1)
            self.pub_target = ros.Publisher(self.target_pub_name,
                                            Float64MultiArray,
                                            queue_size=1)
            self.pub_mix = ros.Publisher(self.mix_pub_name,
                                         Float64MultiArray,
                                         queue_size=1)
        if self.publish_loss:
            self.pub_loss = ros.Publisher(self.loss_pub_name,
                                          Float64,
                                          queue_size=1)
            self.pub_acc = ros.Publisher(self.acc_pub_name,
                                         Float64,
                                         queue_size=1)
        if self.sm:
            self.ser_sm = ros.Service(self.sm_ser_name, Trigger, self._sm_transition)

    def get_actions(self, state, pred=True):

        # Load target action
        if self.play_from_file:
            target = self.network.interpolate(self.target_fct, self.t).tolist()[0]
        else:
            target = self.physics.get_hyq_tgt_action().tolist()[0]
            self.last_action_it = self.physics.hyq_action_it

        # If we need the prediction
        if pred:
            # Predict network action
            if len(state) == 24:
                predicted = self.network.predict(np.mat(state)).tolist()[0]
            else:
                predicted = []

        else:
            predicted = []

        return predicted, target

    def get_states(self):

        curr_state = self.physics.get_hyq_state().tolist()[0]
        self.last_state_it = self.physics.hyq_state_it

        if self.play_from_file:
            rec_state_1 = self.network.interpolate(self.state_1_fct, self.t)

            if hasattr(self, 'state_2_fct'):
                rec_state_2 = self.network.interpolate(self.state_2_fct, self.t)
                rec_state = np.mat(np.hstack((rec_state_1, rec_state_2))).tolist()[0]
            else:
                rec_state = rec_state_1.tolist()[0]
        else:
            rec_state = [0] * len(curr_state)

        return curr_state, rec_state

    def _training_thread(self, x, y):

        if self.verbose >= 1:
            print " [Training]  It: " + str(self.it) + \
                  "\tRCF It: " + str(self.last_action_it) + \
                  " - Fitting NN with FV of shape " + \
                  str(x.shape) + " x " + str(y.shape) + \
                  " through " + str(self.epoch_num) + " epochs!"
        self.loss, self.accuracy = self.network.train(x, y, n_epochs=self.epoch_num,  # plot_train_states=True,
                                                      evaluate=False, save=False)
        self.train_it += 1
        if self.verbose >= 1:
            print " [Training]  It: " + str(self.it) + \
                  "\tRCF It: " + str(self.last_action_it) + \
                  " - Finished with Loss: {:.5f}".format(self.loss)

    def train_run_sm_step(self):

        # GET DATA (ALWAYS)
        curr_state, rec_state = self.get_states()
        pred_action, tgt_action = self.get_actions(curr_state, pred=(self.train_it > 0))

        # TRAINING MODE
        if self.train_sm_mode == "Training":

            if self.train:
                # Record states and action in a buffer
                if len(curr_state) == 24 and len(tgt_action) == 8:
                    self.x_train_step.append(curr_state)
                    self.y_train_step.append(tgt_action)

                # When no training yet done
                if self.training_thread is None:
                    # When buffer is full, train
                    if len(self.x_train_step) == self.train_buff_size:
                        x = np.mat(self.x_train_step[-self.train_buff_size:])
                        y = np.mat(self.y_train_step[-self.train_buff_size:])
                        self.training_thread = threading.Thread(name="train_thread", target=self._training_thread,
                                                                args=[x, y])
                        # Hack to avoid racing condition between training and execution thread
                        self.training_thread.start()
                        self.x_train_step = []
                        self.y_train_step = []
                else:
                    # When buffer is full
                    if len(self.x_train_step) > self.train_buff_size:
                        # When training is finished start again with new buffer
                        if not self.training_thread.isAlive():
                            x = np.mat(self.x_train_step)  # [-self.train_buff_size:])
                            y = np.mat(self.y_train_step)  # [-self.train_buff_size:])
                            self.training_thread = threading.Thread(target=self._training_thread,
                                                                    args=[x, y])
                            # Hack to avoid racing condition between training and execution thread
                            self.training_thread.start()
                            self.x_train_step = []
                            self.y_train_step = []
                            self.train_it += 1

        # TRANSITION TRAINING -> PREDICTION
        elif self.train_sm_mode == "Training_running_transition":
            self.x_train_step = []
            self.y_train_step = []
            self.nn_weight = 1

            if self.training_thread is not None:
                while self.training_thread.isAlive():
                    if hasattr(self.network.readout, "model"):
                        self.network.readout.model.stop_training = True
                    time.sleep(0.1)
            print "\n [State Machine] Transition to Full NN control by-passing RCF at action it: " + \
                  str(self.last_action_it) + "!"
            self.train_sm_mode = "Running"

        # PREDICTION MODE
        elif self.train_sm_mode == "Running":
            pass

        # TRANSITION PREDICTION -> TRAINING
        elif self.train_sm_mode == "Running_training_transition":
            self.nn_weight = 0
            print "\n [State Machine] Transition to RCF control and NN Training!"
            self.train_sm_mode = "Training"

        # SEND PRED ON ROS (ALWAYS)
        if len(pred_action) == 8:
            self.physics.send_hyq_nn_pred(pred_action, self.nn_weight, np.array(tgt_action) - np.array(pred_action))

        # DEBUG AND LOGGING (ALWAYS)
        # Publish data on ROS topic to debug
        if self.nn_weight == 1:
            if len(pred_action) > 0:
                mix_action = pred_action
            else:
                mix_action = tgt_action
        else:
            mix_action = tgt_action
        self._publish_states(curr_state, rec_state)
        self._publish_actions(pred_action, tgt_action, mix_action)
        self._publish_loss()

        # Save states and actions
        if self.save_ctrl:
            self.save_ctrl_state.append(curr_state)
            self.save_ctrl_action.append(mix_action)

    def train_step(self):

        # Get state and target action
        curr_state, rec_state = self.get_states()
        pred_action, tgt_action = self.get_actions(curr_state, pred=(self.train_it > 0))

        # TRAINING
        # Always continue filling the buffer
        if len(curr_state) != 0 and len(tgt_action) != 0:
            self.x_train_step.append(curr_state)
            self.y_train_step.append(tgt_action)
            if len(self.x_train_step) > self.train_buff_size:
                self.x_train_step.pop(0)
                self.y_train_step.pop(0)

        # When no training yet done
        if self.training_thread is None:
            # When buffer is full, train
            if len(self.x_train_step) == self.train_buff_size:
                x = np.mat(self.x_train_step)
                y = np.mat(self.y_train_step)
                self.training_thread = threading.Thread(target=self._training_thread, args=[x, y])
                self.training_thread.start()
                self.x_train_step = []
                self.y_train_step = []
        else:

            # When buffer is full
            if len(self.x_train_step) >= self.train_buff_size:
                # When training is finished start again with new buffer
                if not self.training_thread.isAlive():
                    x = np.mat(self.x_train_step)
                    y = np.mat(self.y_train_step)
                    self.training_thread = threading.Thread(target=self._training_thread, args=[x, y])
                    self.training_thread.start()
                    self.x_train_step = []
                    self.y_train_step = []

        # EXECUTION
        # Define the weight between NN prediction and RCF Controller
        mix_action = tgt_action
        self.nn_weight = 0

        if not self.nn_started and len(pred_action) == 8:
            print(" [Execution] It: " + str(self.it) +
                  "\tNN has started publishing meaningful values! ")
            self.nn_started = True

        if not self.ol and len(pred_action) == 8:
            if self.t > self.t_stop_cl:
                self.nn_weight = 1
                mix_action = np.array(pred_action)

            elif self.t > self.t_start_cl:
                self.nn_weight = (self.t - self.t_start_cl) / self.t_cl

                # Compute the sum action for debug purpose
                mix_action = ((1 - self.nn_weight) * np.array(tgt_action)) + \
                             (self.nn_weight * np.array(pred_action))

        # Send the NN prediction to the RCF controller
        if len(pred_action) == 8:
            self.physics.send_hyq_nn_pred(pred_action, self.nn_weight,
                                          np.array(tgt_action) - np.array(pred_action))

        # Publish data on ROS topic to debug
        self._publish_states(curr_state, rec_state)
        self._publish_actions(pred_action, tgt_action, mix_action)
        self._publish_loss()

        # Save states and actions
        if self.save_ctrl:
            self.save_ctrl_state.append(curr_state)
            self.save_ctrl_action.append(mix_action)

    def step(self):

        # Get state and target action
        curr_state, rec_state = self.get_states()
        pred_action, tgt_action = self.get_actions(curr_state, pred=(self.train_it > 0))

        # Define the weight between NN prediction and RCF Controller
        mix_action = tgt_action
        self.nn_weight = 0

        if not self.ol and len(pred_action) == 8:
            if self.t > self.t_start_cl:
                self.nn_weight = 1
                mix_action = np.array(pred_action)

        # Send the NN prediction to the RCF controller
        if len(pred_action) == 8:
            self.physics.send_hyq_nn_pred(pred_action,
                                          self.nn_weight,
                                          np.array(tgt_action) -
                                          np.array(pred_action))

        # Publish data on ROS topic to debug
        self._publish_states(curr_state, rec_state)
        self._publish_actions(pred_action, tgt_action, mix_action)
        self._publish_loss()

        # Save states and actions
        if self.save_ctrl:
            self.save_ctrl_state.append(curr_state)
            self.save_ctrl_action.append(mix_action)

    def run(self):

        self.start()

        self.printv("\n ===== Running the Simulation Loop =====\n")

        last_it = 0
        last_t = 0
        step_it = 0
        step_t = 0
        t_init = time.time()
        trot_flag = False
        while not ros.is_shutdown() and self.t < self.t_sim:

            self.t = self.physics.get_sim_time()
            if self.it % 1000 == 0:
                step_it = 0
                step_t = self.t

            if self.t > 1:

                # Plot the curves if necessary
                if self.train_it == 1:
                    if self.plot:
                        self._start_plotter()

                # Start trotting when everything initialized
                if not self.remote and not trot_flag:
                    trot_flag = self.physics.start_rcf_trot()
                    if trot_flag:
                        self.printv(" ===== Trotting Started =====\n")
                        if self.save_states or self.save_metrics:
                            self.save_states_t_trot = copy.copy(self.t)
                            self.save_trot_i = self.save_index

                # Apply noise on the robot
                # if not self.remote and trot_flag:
                #     self.physics.apply_noise()

                # Choose execution mode
                if self.sm:
                    self.train_run_sm_step()
                else:
                    if self.t < self.t_train:
                        self.train_step()
                    else:
                        self.step()

                # Display simulation progress
                if last_it == 0:
                    last_it = self.it
                if self.it % 500 == 0 and self.verbose >= 1:
                    tp = (self.t - last_t)
                    if tp != 0:
                        f = (self.it - last_it) / tp
                        print(" [Execution] It: " + str(self.it) +
                              "\tSim Time: " + "{:.2f}".format(self.t) + " s" +
                              "\tNN Freq: " + "{:.2f}".format(f) + " Hz" +
                              " / Weight: "+"{:.2f}".format(self.nn_weight) +
                              "\tRobot dist: {:.2f} m".format(math.sqrt(sum([float(i)**2
                                                                             for i in self.physics.get_hyq_x_y_z()]))))
                        last_it = self.it
                        last_t = self.t

                # Save physics states and metrics
                self.t_hist.append(self.t)
                if self.save_states or self.save_metrics:
                    curr_x, curr_y, curr_z = self.physics.get_hyq_x_y_z()
                    curr_phi, curr_theta, curr_psi = self.physics.get_hyq_phi_theta_psi()
                    self.save_states_t_real.append(time.time() - t_init)
                    self.save_states_x.append(curr_x)
                    self.save_states_y.append(curr_y)
                    self.save_states_z.append(curr_z)
                    self.save_states_phi.append(curr_phi)
                    self.save_states_theta.append(curr_theta)
                    self.save_states_psi.append(curr_psi)
                    self.save_index += 1
                    if self.nn_weight > 0 and self.save_stop_train_i == 0:
                        self.save_stop_train_i = self.save_index
                    if self.nn_weight >= 1.0 and self.save_start_test_i == 0:
                        self.save_start_test_i = self.save_index

            # Pause
            self.it += 1
            step_it += 1
            while self.physics.get_sim_time() < step_t + step_it * self.time_step:
                time.sleep(0.0001)

        self.printv(" ===== Final robot distance: "
                    "{:.2f} m =====\n".format(math.sqrt(sum([float(i)**2 for i in self.physics.get_hyq_x_y_z()]))))
        self.stop()

    def stop(self):

        # Stop the physics thread
        self.physics.stop()

        # Save the states and actions (IOs of the controller)
        if self.save_ctrl:
            with open(self.folder + "/ctrl.pkl", "wb") as f:
                pickle.dump([np.mat(self.save_ctrl_state), np.mat(self.save_ctrl_action), np.array(self.t_hist)],
                            f, protocol=2)

            if self.sim_file is not None and not self.ol:
                self.network.save()

        # Save the physics states
        if self.save_states:
            to_save = {"t_sim": self.t_hist, "t_real": self.save_states_t_real, "t_trot": self.save_states_t_trot,
                       "x": self.save_states_x, "y": self.save_states_y, "z": self.save_states_z,
                       "phi": self.save_states_phi, "theta": self.save_states_theta, "psi": self.save_states_psi}

            pickle.dump(to_save, open(self.folder + "/states.pkl", "wb"), protocol=2)

        # Save the robot metrics
        if self.save_metrics:
            if self.t_train > 0 and not self.ol and self.t_start_cl < self.t_sim:
                (r_f, r_train_fft, r_test_fft, r_rms) = self._compute_diff_fft_sig(self.t_hist[self.save_trot_i:],
                                                                                   self.save_states_psi[self.save_trot_i:],
                                                                                   self.save_stop_train_i,
                                                                                   self.save_start_test_i)
                (p_f, p_train_fft, p_test_fft, p_rms) = self._compute_diff_fft_sig(self.t_hist[self.save_trot_i:],
                                                                                   self.save_states_phi[self.save_trot_i:],
                                                                                   self.save_stop_train_i,
                                                                                   self.save_start_test_i)
                to_save = {"train_roll_range": max(self.save_states_phi[self.save_trot_i:self.save_stop_train_i]) -
                                               min(self.save_states_phi[self.save_trot_i:self.save_stop_train_i]),
                           "train_pitch_range": max(self.save_states_psi[self.save_trot_i:self.save_stop_train_i]) -
                                                min(self.save_states_psi[self.save_trot_i:self.save_stop_train_i]),
                           "train_x_range": self.save_states_x[self.save_stop_train_i] -
                                            self.save_states_x[self.save_trot_i],
                           "train_y_range": abs(self.save_states_y[self.save_stop_train_i] -
                                                self.save_states_y[self.save_trot_i]),
                           "cl_roll_range": max(self.save_states_phi[self.save_stop_train_i:self.save_start_test_i]) -
                                            min(self.save_states_phi[self.save_stop_train_i:self.save_start_test_i]),
                           "cl_pitch_range": max(self.save_states_psi[self.save_stop_train_i:self.save_start_test_i]) -
                                             min(self.save_states_psi[self.save_stop_train_i:self.save_start_test_i]),
                           "cl_x_range": self.save_states_x[self.save_start_test_i] -
                                         self.save_states_x[self.save_stop_train_i],
                           "cl_y_range": abs(self.save_states_y[self.save_start_test_i] -
                                             self.save_states_y[self.save_stop_train_i]),
                           "test_roll_range": max(self.save_states_phi[self.save_start_test_i:]) -
                                              min(self.save_states_phi[self.save_start_test_i:]),
                           "test_pitch_range": max(self.save_states_psi[self.save_start_test_i:]) -
                                               min(self.save_states_psi[self.save_start_test_i:]),
                           "test_x_range": self.save_states_x[-1] - self.save_states_x[self.save_start_test_i],
                           "test_y_range": abs(self.save_states_y[-1] - self.save_states_y[self.save_start_test_i]),
                           "pitch_fft_rms": p_rms, "roll_fft_rms": r_rms,
                           "t_train": self.t_hist[self.save_stop_train_i] - self.t_hist[self.save_trot_i],
                           "t_cl": self.t_hist[self.save_start_test_i] - self.t_hist[self.save_stop_train_i],
                           "t_test": self.t_hist[-1] - self.t_hist[self.save_start_test_i]
                           }
            else:
                to_save = {"roll_range": max(self.save_states_phi[self.save_trot_i:]) -
                                         min(self.save_states_phi[self.save_trot_i:]),
                           "pitch_range": max(self.save_states_psi[self.save_trot_i:]) -
                                          min(self.save_states_psi[self.save_trot_i:]),
                           "x_range": self.save_states_x[-1] - self.save_states_x[self.save_trot_i],
                           "y_range": abs(self.save_states_y[-1] - self.save_states_y[self.save_trot_i]),
                           "t_sim": self.t_hist[-1] - self.t_hist[self.save_trot_i]
                           }

            pickle.dump(to_save, open(self.folder + "/metrics.pkl", "wb"), protocol=2)

        # Stop plotjuggler
        if self.plot:
            self._stop_plotter()

    def _compute_diff_fft_sig(self, t, sig, ind_stop_train, ind_start_test):

        # Find and interpolate vectors before and after training
        sig1_fn = interp1d(t[0:ind_stop_train], sig[0:ind_stop_train])
        sig2_fn = interp1d(t[ind_start_test:], sig[ind_start_test:])
        t_max = min(t[ind_stop_train] - t[0], t[-1] - t[ind_start_test])
        t_1 = np.linspace(t[20], t[0] + t_max - 1, 10000)
        t_2 = np.linspace(t[ind_start_test+20], t[ind_start_test] + t_max - 1, 10000)
        sig1 = sig1_fn(t_1)
        sig2 = sig2_fn(t_2)

        # Compute the two FFTs
        ts = t_1[1] - t_1[0]
        fs = 1.0/ts
        n = len(sig1)
        k = np.arange(n)
        period = n / fs
        frq = k / period
        frq = frq[range(n/2)]
        sig1_fft = np.fft.fft(sig1) / n
        sig1_fft = sig1_fft[range(n/2)]
        sig2_fft = np.fft.fft(sig2) / n
        sig2_fft = sig2_fft[range(n/2)]

        f_max_eval = int(20 * period)  # 20 Hz
        # plt.plot(frq[0:f_max_eval], abs(sig1_fft[0:f_max_eval]), label="RCF sig FFT")
        # plt.plot(frq[0:f_max_eval], abs(sig2_fft[0:f_max_eval]), label="NN sig FFT")
        # plt.xlabel('Freq (Hz)')
        # plt.ylabel('|Y(freq)|')
        # plt.legend()
        # plt.show()

        # Compute RMS difference
        fft_rms_diff = 0
        for (x, y) in zip(abs(sig1_fft[0:f_max_eval]), abs(sig2_fft[0:f_max_eval])):
            fft_rms_diff += (x - y) ** 2

        # print "The RMS difference between the two FFT is: " + str(fft_rms_diff)

        return (frq[0:f_max_eval],
                abs(sig1_fft[0:f_max_eval]),
                abs(sig2_fft[0:f_max_eval]),
                math.sqrt(fft_rms_diff / len(sig1_fft[0:f_max_eval])))

    def _start_plotter(self):

        # Do not start several times
        if self.plot_ps is not None:
            if self.plot_ps.isalive():
                return

        # Create plotjuggler process
        proc = "rosrun plotjuggler PlotJuggler -n --buffer_size 9 -l plot.xml"

        self.printv("\n ===== Starting PlotJuggler =====\n")

        self.plot_ps = pexpect.spawn(proc)
        if self.verbose == 2:
            self.plot_ps.logfile_read = sys.stdout

    def _stop_plotter(self):

        if self.plot_ps is not None:
            self.printv("\n\n ===== Stopping PlotJuggler  =====\n")
            self.plot_ps.sendcontrol('c')

            # Kill the process if not finished correctly
            for proc in psutil.process_iter():
                name = " ".join(proc.cmdline())
                if self.ps_to_kill in name:
                    proc.kill()

    def _publish_actions(self, pred, target, action):

        if self.publish_actions:
            if pred is not None:
                pred_array = Float64MultiArray(data=pred)
                self.pub_pred.publish(pred_array)
            if target is not None:
                target_array = Float64MultiArray(data=target)
                self.pub_target.publish(target_array)
            if action is not None:
                mix_array = Float64MultiArray(data=action)
                self.pub_mix.publish(mix_array)

    def _publish_states(self, curr_state, rec_state):

        if self.publish_states:
            if curr_state is not None:
                curr_arr = Float64MultiArray(data=curr_state)
                self.pub_curr_state.publish(curr_arr)
            if rec_state is not None:
                rec_arr = Float64MultiArray(data=rec_state)
                self.pub_rec_state.publish(rec_arr)

    def _publish_loss(self):

        self.pub_loss.publish(Float64(self.loss))
        self.pub_acc.publish(Float64(self.accuracy))

    def _sm_transition(self, msg):

        ack = None
        if self.train_sm_mode == "Training":
            self.train_sm_mode = "Training_running_transition"
            ack = "Switched to Running Mode!"
        elif self.train_sm_mode == "Running":
            self.train_sm_mode = "Running_training_transition"
            ack = "Switched to Training Mode!"

        if ack is not None:
            return TriggerResponse(success=True, message=ack)
        else:
            return TriggerResponse(success=False, message="Cannot trigger transition, please check the NN process!")

    def printv(self, txt):
        if self.verbose >= 1:
            print(txt)


class CPGSimulation(Simulation):

    def __init__(self, folder=None):

        super(CPGSimulation, self).__init__(folder=folder)
        self.cpg_config = {"params": [{'mu': 1, 'omega': 6.35, 'duty_factor': 0.5,
                                       'phase_offset': 0, 'o': 0, 'r': 1,
                                       'coupling': [0, -1, -1, 1]},
                                      {'mu': 1, 'omega': 6.35, 'duty_factor': 0.5,
                                       'phase_offset': 1, 'o': 0, 'r': 1,
                                       'coupling': [-1, 0, 1, -1]},
                                      {'mu': 1, 'omega': 6.35, 'duty_factor': 0.5,
                                       'phase_offset': 3, 'o': 0, 'r': 1,
                                       'coupling': [-1, 1, 0, -1]},
                                      {'mu': 1, 'omega': 6.35, 'duty_factor': 0.5,
                                       'phase_offset': 0, 'o': 0, 'r': 1,
                                       'coupling': [1, -1, -1, 0]}],
                           "integ_time": 0.001}
        self.cpg = None

    def start(self):

        # Create and start the handle thread to the physics simulation
        self.physics = physics.HyQSim(view=self.view, verbose=self.verbose)
        self.physics.start()
        self.physics.register_node()

        # Create and load the trained network
        self.cpg = control.CPG(self.cpg_config)

        # Create the ROS topics to debug simulation
        if self.publish_actions:
            self.pub_pred = ros.Publisher(self.pred_pub_name,
                                          Float64MultiArray,
                                          queue_size=50)
            self.pub_target = ros.Publisher(self.target_pub_name,
                                            Float64MultiArray,
                                            queue_size=50)
            self.pub_mix = ros.Publisher(self.mix_pub_name,
                                         Float64MultiArray,
                                         queue_size=50)

        # Create history arrays
        self.save_ctrl_state = []
        self.save_ctrl_action = []

    def step(self):

        # Get time
        self.t = self.physics.get_sim_time()

        if self.t > 1:

            # Fill time history
            self.t_hist.append(self.t)

            # Get last state
            state = self.physics.get_hyq_state()

            # Get CPG motor action
            action = np.zeros(24)
            r = np.sin(2*np.pi*2*self.t)
            # , a in enumerate(self.cpg.step(self.t
            action[2] = r
            action[5] = r
            action[8] = -r
            action[11] = -r
            predicted = None
            target = action

            # Send action
            self.physics.set_hyq_action(action)
            self.physics.send_hyq_traj()

            # Publish data on ROS topic to debug
            self._publish_states(state, state)
            self._publish_actions(predicted, target, action)

            # Save states and actions
            if self.save_ctrl:
                self.save_ctrl_state.append(state.tolist()[0])
                self.save_ctrl_action.append(action.tolist())


if __name__ == '__main__':

    if len(sys.argv) > 1:

        s = Simulation(folder=sys.argv[1])
        s.run()

    else:
        print("\nFailure: simulation.py expects a folder name in argument\n")
        exit(-1)