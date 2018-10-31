from collections import deque
import copy
from keras import backend as K
import numpy as np
import pexpect
import pickle
import psutil
import rospy as ros
import rosbag
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

    def __init__(self, nn_folder=None, sim_file=None, verbose=2, t_train=0,
                 pub_actions=True, publish_states=True, t_sim=180, t_start_cl=15, sm=False,
                 t_stop_cl=160, save_folder=None, ol=False, view=False, train=True,
                 pub_loss=True, epoch_num=150, plot=False, pub_error=True,
                 init_impedance=None, train_buff_size=10000, real_physics=False):

        self.t_sim = t_sim
        self.t_train = t_train
        self.t_start_cl = t_start_cl
        self.t_stop_cl = t_stop_cl
        self.t_cl = self.t_stop_cl - self.t_start_cl

        # Class variable
        self.view = view
        self.plot = plot
        self.verbose = verbose
        self.nn_folder = nn_folder
        self.publish_actions = pub_actions
        self.publish_states = publish_states
        self.publish_loss = pub_loss
        self.publish_error = pub_error
        self.save_folder = save_folder
        self.sim_file = sim_file
        self.ol = ol
        self.epoch_num = epoch_num
        self.train_buff_size = train_buff_size
        self.real_physics = real_physics
        self.init_impedance = init_impedance
        self.sm = sm
        self.train = train
        self.play_from_file = False

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
        self.train_session = None
        self.pred_session = None
        self.train_graph = None
        self.pred_graph = None
        self.train_pred_lock = threading.Lock()

        # Training variable
        self.t_hist = [0]
        self.state = deque([])
        self.state_history = []
        self.action_history = []
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

    def start(self):

        # Create and start the handle thread to the physics simulation
        self.physics = physics.HyQSim(init_impedance=None,
                                      view=self.view,
                                      remote=self.real_physics,
                                      verbose=self.verbose,
                                      rt=self.real_physics,
                                      publish_error=self.publish_error)
        ros.init_node("simulation", anonymous=True)
        self.physics.start()

        # If sim folder is specified create a untrained network and read simulation from a FILE
        if self.sim_file is not None:
            self.play_from_file = True
            self.network = network.NN(data_file=self.sim_file,
                                      save_folder=self.save_folder,
                                      batch_size=256,
                                      val_split=0.01,
                                      verbose=0)
            self.network.load_data(True)

        # Else, load the trained network and start a REAL simulation
        elif self.nn_folder is not None:
            self.play_from_file = False
            self.network = network.NN(max_epochs=100000,
                                      checkpoint=False,
                                      esn_n_read=10,
                                      no_callbacks=True,
                                      esn_in_mask=[True, False, False, False, False],
                                      esn_out_mask=[True] * 24,
                                      esn_real_fb=True,
                                      verbose=0)
            self.network.load(self.nn_folder, load_all=True)
            self.network.esn.verbose = 0
            self.train_it += 1

        else:
            self.play_from_file = False
            self.network = network.NN(max_epochs=100000,
                                      checkpoint=False,
                                      esn_n_read=10,
                                      no_callbacks=True,
                                      esn_in_mask=[True, False, False, False, False],
                                      esn_out_mask=[False] * 24,
                                      verbose=0,
                                      save_folder=self.save_folder)

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
            if len(state) == 5:
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

        if self.verbose > 1:
            print " [Training] A Iteration: " + str(self.last_action_it) + \
                  "\tFitting ESN with feature vectors of shape " + \
                  str(x.shape) + " x " + str(y.shape) + "!"
        self.loss, self.accuracy = self.network.train(x, y, n_epochs=self.epoch_num,  # plot_train_states=True,
                                                      evaluate=False, save=False)
        if self.verbose > 1:
            print " [Training] A Iteration: " + str(self.last_action_it) + \
                  "\tFinished with Loss: {:.5f}".format(self.loss)

    def train_run_sm_step(self):

        # GET DATA (ALWAYS)
        curr_state, rec_state = self.get_states()
        pred_action, tgt_action = self.get_actions(curr_state, pred=(self.train_it > 0)) #(self.train_sm_mode == "Running"))

        # TRAINING MODE
        if self.train_sm_mode == "Training":

            if self.train:
                # Record states and action in a buffer
                if len(curr_state) == 5 and len(tgt_action) == 24:
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
                            x = np.mat(self.x_train_step) # [-self.train_buff_size:])
                            y = np.mat(self.y_train_step) #[-self.train_buff_size:])
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
        if len(pred_action) == 24:
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
        if self.save_folder is not None:
            self.state_history.append(curr_state)
            self.action_history.append(mix_action)

    def train_step(self):

        # Get state and target action
        curr_state, rec_state = self.get_states()
        pred_action, tgt_action = self.get_actions(curr_state, pred=(self.train_it > 0))

        # TRAINING
        # Always continue filling the buffer
        if len(curr_state) != 0 and len(tgt_action) != 0:
            self.x_train_step.append(curr_state)
            self.y_train_step.append(tgt_action)

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
            if len(self.x_train_step) > self.train_buff_size:

                # When training is finished start again with new buffer
                if not self.training_thread.isAlive():
                    x = np.mat(self.x_train_step)
                    y = np.mat(self.y_train_step)
                    self.training_thread = threading.Thread(target=self._training_thread, args=[x, y])
                    self.training_thread.start()
                    self.x_train_step = []
                    self.y_train_step = []
                    self.train_it += 1

        # EXECUTION
        # Define the weight between NN prediction and RCF Controller
        mix_action = tgt_action
        self.nn_weight = 0

        if not self.ol and len(pred_action) == 24:
            if self.t > self.t_stop_cl:
                self.nn_weight = 1
                mix_action = np.array(pred_action)

            elif self.t > self.t_start_cl:
                self.nn_weight = (self.t - self.t_start_cl) / self.t_cl

                # Compute the sum action for debug purpose
                mix_action = ((1 - self.nn_weight) * np.array(tgt_action)) + \
                             (self.nn_weight * np.array(pred_action))

        # Send the NN prediction to the RCF controller
        if len(pred_action) == 24:
            self.physics.send_hyq_nn_pred(pred_action, self.nn_weight,
                                          np.array(tgt_action) - np.array(pred_action))

        # Publish data on ROS topic to debug
        self._publish_states(curr_state, rec_state)
        self._publish_actions(pred_action, tgt_action, mix_action)
        self._publish_loss()

        # Save states and actions
        if self.save_folder is not None:
            self.state_history.append(curr_state)
            self.action_history.append(mix_action)

    def step(self):

        # Get state and target action
        curr_state, rec_state = self.get_states()
        pred_action, tgt_action = self.get_actions(curr_state, pred=(self.train_it > 0))

        # Define the weight between NN prediction and RCF Controller
        mix_action = tgt_action
        self.nn_weight = 0

        if not self.ol and len(pred_action) == 24:
            if self.t > self.t_start_cl:
                self.nn_weight = 1
                mix_action = np.array(pred_action)

        # Send the NN prediction to the RCF controller
        if len(pred_action) == 24:
            self.physics.send_hyq_nn_pred(pred_action,
                                          self.nn_weight,
                                          np.array(tgt_action) -
                                          np.array(pred_action))

        # Publish data on ROS topic to debug
        self._publish_states(curr_state, rec_state)
        self._publish_actions(pred_action, tgt_action, mix_action)
        self._publish_loss()

        # Save states and actions
        if self.save_folder is not None:
            self.state_history.append(curr_state)
            self.action_history.append(mix_action)

    def run(self):

        self.start()

        self.printv("\n\n ===== Running the Simulation Loop =====\n")

        last_it = 0
        last_t = 0
        trot_flag = False
        while not ros.is_shutdown() and self.t < self.t_sim:

            self.t = self.physics.get_sim_time()

            if self.t > 1:

                # Plot the curves if necessary
                if self.train_it == 1:
                    if self.plot:
                        self._start_plotter()

                # Start trotting when everything initialized
                if not self.real_physics and not trot_flag:
                    self.physics.start_rcf_trot()
                    trot_flag = True

                self.t_hist.append(self.t)

                # Apply noise on the robot
                if not self.real_physics and trot_flag:
                    self.physics.apply_noise()

                # Choose execution mode
                if self.sm:
                    self.train_run_sm_step()
                else:
                    if self.t < self.t_train:
                        self.train_step()
                    else:
                        self.step()

                # Display simulation progress
                if self.it % 1000 == 0 and self.verbose > 1:
                    tp = (self.t - last_t)
                    if tp != 0:
                        f = (self.it - last_it) / tp
                        print(" [Execution] Iteration: " + str(self.it) +
                              "\tSim Time: " + "{:.2f}".format(self.t) + " s" +
                              "\tNN Freq: " + "{:.2f}".format(f) + " Hz" +
                              "\tNN Weight: "+"{:.2f}".format(self.nn_weight))
                        last_it = self.it
                        last_t = self.t

                self.it += 1

        self.stop()

    def stop(self):

        # Save new simulation in OL
        if self.save_folder is not None:
            with open(self.save_folder + "/state_action.pkl", "wb") as f:
                pickle.dump([np.mat(self.state_history), np.mat(self.action_history), np.array(self.t_hist)],
                            f, protocol=2)
            if self.sim_file is not None and not self.ol:
                self.network.save()

        if self.plot:
            self._stop_plotter()

        self.physics.stop()

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

    def __init__(self):

        super(CPGSimulation, self).__init__(nn_folder=None, view=True, ol=True, pub_actions=True)
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
        self.state_history = []
        self.action_history = []

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
            if self.save_folder is not None:
                self.state_history.append(state.tolist()[0])
                self.action_history.append(action.tolist())


if __name__ == '__main__':

    if len(sys.argv) > 1:

        if sys.argv[1] == "ol":
            folder = "data/nn_sim_learning/" + utils.timestamp()
            utils.mkdir(folder)
            s = Simulation(sim_file=sys.argv[2], view=True, ol=True,
                           pub_actions=True, save_folder=folder)
            s.run()

        if sys.argv[1] == "cl":
            folder = "data/nn_sim_learning/" + utils.timestamp()
            utils.mkdir(folder)
            s = Simulation(nn_folder=sys.argv[2], save_folder=folder,
                           t_sim=1000, t_train=999, t_start_cl=1500,
                           t_stop_cl=1600)
            s.run()

        if sys.argv[1] == "train":
            folder = "data/nn_sim_learning/" + utils.timestamp()
            utils.mkdir(folder)
            s = Simulation(save_folder=folder, view=False, plot=False,
                           t_sim=500, t_train=150, t_start_cl=40,
                           t_stop_cl=800)
            s.run()

        if sys.argv[1] == "cpg":
            s = CPGSimulation()
            s.run()

        if sys.argv[1] == "hyq":
            folder = "data/nn_sim_learning/" + utils.timestamp()
            utils.mkdir(folder)
            s = Simulation(save_folder=folder, plot=False, train=True,
                           t_sim=10000, sm=True, real_physics=True)
            s.run()

        if sys.argv[1] == "hyq_pretrained":
            folder = "data/nn_sim_learning/" + utils.timestamp()
            utils.mkdir(folder)
            s = Simulation(nn_folder=sys.argv[2], save_folder=folder,
                           view=False, plot=True, train=True,
                           t_sim=10000, sm=True, real_physics=True)
            s.run()