from collections import deque
import numpy as np
import pickle
import rospy as ros
import rosbag
import sys
import time

from std_msgs.msg import Float64MultiArray, MultiArrayLayout

import control
import physics
import network
import utils


class Simulation(object):

    def __init__(self, nn_folder=None, sim_file=None, verbose=2, t_pretrain=0,
                 pub_actions=True, pub_states=True, t_sim=180, t_start_cl=15,
                 t_stop_cl=160, save_folder=None, ol=False, view=False):

        self.t_sim  = t_sim
        self.t_pretrain = t_pretrain
        self.t_start_cl = t_start_cl
        self.t_stop_cl = t_stop_cl
        self.t_cl = self.t_stop_cl - self.t_start_cl

        self.view = view
        self.verbose = verbose
        self.nn_folder = nn_folder
        self.publish_actions = pub_actions
        self.publish_states = pub_states
        self.save_folder = save_folder
        self.sim_file = sim_file
        self.ol = ol

        self.t = 0
        self.it = 0
        self.action_it = 0
        self.train_it = 0

        self.t_hist = [0]
        self.state = deque([])
        self.state_history = []
        self.action_history = []
        self.x_train_step = []
        self.y_train_step = []

        self.curr_state_pub_name = "/sim_debug/curr_states"
        self.rec_state_pub_name = "/sim_debug/rec_states"
        self.pred_pub_name = "/sim_debug/prediction"
        self.target_pub_name = "/sim_debug/target"
        self.mix_pub_name = "/sim_debug/mix_cl"

    def start(self):

        # Create history arrays

        # Create and start the handle thread to the physics simulation
        self.physics = physics.HyQSim(view=self.view, verbose=self.verbose)
        self.physics.start()
        self.physics.register_node()

        # If sim folder is specified create a NOT-trained network
        if self.sim_file is not None:
            self.play_from_sim = True
            folder = "data/nn_sim_learning/"+ utils.timestamp()
            utils.mkdir(folder)
            self.network = network.FeedForwardNN(
                                       data_file=self.sim_file,
                                       save_folder=self.save_folder,
                                       batch_size=256,
                                       max_epochs=3,
                                       val_split=0.01,
                                       verbose=0)
            self.network.load_data(True)

        # Else, load the trained network
        elif self.nn_folder is not None:
            self.play_from_sim = True
            self.network = network.FeedForwardNN(verbose=self.verbose)
            self.network.load(self.nn_folder, load_all=True)
            self.network.verbose = self.verbose
            self.network.save_folder = self.save_folder

        else:
            self.play_from_sim = False
            self.network = network.FeedForwardNN(verbose=self.verbose,
                                                save_folder=self.save_folder)

        # Retrieve interpolation functions for the target and del the rest
        if self.play_from_sim:
            self.target_fct = self.network.get_fct(self.network.y_t,
                                                   self.network.y_val)
            self.state_1_fct = self.network.get_fct(self.network.x_t,
                                                   self.network.x_val)
            if self.network.x2_t.shape[0] is not 0:
                self.state_2_fct = self.network.get_fct(self.network.x2_t,
                                                        self.network.x2_val)

            if max(self.network.y_t) - 1 < self.t_sim:
                self.printv("\nThe nn data recorded file is too short. Lowering the simulation time from {:.2f}s to ".format(self.t_sim) + \
                    "{:.2f}s".format(max(self.network.y_t) - 1))
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

    def get_actions(self, state, pred=True):

        # Load target action
        if self.play_from_sim:
            target = self.network.interpolate(self.target_fct, self.t).tolist()[0]
        else:
            target = self.physics.get_hyq_action().tolist()[0]

        # If we need the prediction
        if pred:

            # Predict network action
            if len(state) == 46:

                # self.state.append(state)
                # if len(self.state) < 10:
                #     predicted = []
                # else:
                predicted = self.network.predict(np.mat(state)).tolist()
                #self.state.popleft()
            else:
                predicted = []

        else:
            predicted = []

        return predicted, target

    def get_states(self):

        curr_state = self.physics.get_hyq_state().tolist()[0]

        if self.play_from_sim:
            rec_state_1 = self.network.interpolate(self.state_1_fct, self.t)

            if hasattr(self, 'state_2_fct'):
                rec_state_2 = self.network.interpolate(self.state_2_fct, self.t)
                rec_state = np.mat(np.hstack((rec_state_1, rec_state_2))).tolist()[0]
            else:
                rec_state = rec_state_1.tolist()[0]
        else:
            rec_state = [0] * len(curr_state)

        return curr_state, rec_state

    def train_step(self):

        # Get state and target action
        curr_state, rec_state = self.get_states()
        pred_action, tgt_action = self.get_actions(curr_state,
                        pred=(self.train_it > self.network.batch_size))

        # If buffer is not big enough, just fill it
        if len(curr_state) != 0 and len(tgt_action) !=0:
            self.x_train_step.append(curr_state)
            self.y_train_step.append(tgt_action)

        # When buffer is full, train network
        if len(self.x_train_step) == self.network.batch_size:
            x = np.mat(self.x_train_step)
            y = np.mat(self.y_train_step)
            l, a = self.network.train_step(x, y)
            if self.verbose > 1:
                print "Training batch of data - Loss: " + \
                      "{:.5f}".format(l) + " ; Accuracy: " + \
                      "{:.2f}".format(a)
            self.x_train_step = []
            self.y_train_step = []


        # # Actuate robot with target action
        # self.physics.set_hyq_action(tgt_action)
        # self.physics.send_hyq_traj()

        # Publish data on ROS topic to debug
        self._publish_states(curr_state, rec_state)
        self._publish_actions(pred_action, tgt_action, tgt_action)

        # Save states and actions
        if self.save_folder is not None:
            self.state_history.append(curr_state)
            self.action_history.append(tgt_action)

        self.train_it += 1

    def step(self):

        # Get last state
        state, rec_state = self.get_states()

        if self.ol:
            action = self.network.interpolate(self.target_fct, self.t)
            predicted = None
            target = action

        else:
            # Get action
            predicted, target = self.get_actions(state)

            # If we received an action
            if (predicted is not None) and (target is not None):

                # Mix actions
                if self.t < self.t_start_cl:
                    action = target
                else:
                    action = (1 - ((self.t_start_cl - self.t) / self.t_cl)) * \
                             target + \
                             (self.t_start_cl - self.t) / self.t_cl * predicted
            else:
                action = target

        # Send action
        self.physics.set_hyq_action(action)
        self.physics.send_hyq_traj()

        # Publish data on ROS topic to debug
        self._publish_states(state, rec_state)
        self._publish_actions(predicted, target, action)

        # Save states and actions
        if self.save_folder is not None:
            self.state_history.append(state.tolist()[0])
            self.action_history.append(action.tolist())

    def run(self):

        self.start()

        self.printv("\n\n ===== Running the Simulation Loop =====\n")

        last_it = 0
        last_t = 0
        trot_flag = False
        while not ros.is_shutdown() and self.t < self.t_sim:

            self.t = self.physics.get_sim_time()

            if  self.t > 1:

                # Start trotting when everythin initialized
                if trot_flag == False:
                    self.physics.start_rcf_trot()
                    trot_flag = True

                self.t_hist.append(self.t)

                # Choose between execution step or execution+training step
                if self.t < self.t_pretrain:
                    self.train_step()
                else:
                   self.step()

                # Display simulation progress
                if self.it % 1000 == 0 and self.verbose > 1:
                    f = (self.it - last_it) / (self.t - last_t)
                    print("Iteration: " + str(self.it) + \
                         "\tSim Time: " + "{:.2f}".format(self.t) + " s" + \
                         "\tNN Freq: " + "{:.2f}".format(f) + " Hz")
                    last_it = self.it
                    last_t = self.t

            self.it += 1

        self.stop()

    def stop(self):

        # Save new simulation in OL
        if self.save_folder is not None:
            with open(self.save_folder + "state_action.pkl", "wb") as f:
                pickle.dump([np.mat(self.state_history),
                             np.mat(self.action_history),
                             np.array(self.t_hist)],
                             f, protocol=2)
            if self.sim_file is not None and not self.ol:
                self.network.save()

        self.physics.stop()

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

    def printv(self, txt):

        if self.verbose >= 1:
            print(txt)


class CPGSimulation(Simulation):

    def __init__(self):

        super(CPGSimulation, self).__init__(nn_folder=None, view=True,
                                            ol=True, pub_actions=True,
                                            pub_states=True)
        self.cpg_config = {"params":
            [
                {'mu': 1, 'omega': 6.35, 'duty_factor': 0.5,
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
                 'coupling': [1, -1, -1, 0]},
            ],
                        "integ_time": 0.001}

    def start(self):

        # Create and start the handle thread to the physics simulation
        self.physics = physics.HyQSim(view=self.view, verbose=self.verbose)
        self.physics.start()
        self.physics.register_node()

        # Create and load the trained network
        self.cpg = control.CPG(self.cpg_config)

        # Create the ROS topics to debug simulation
        if self.publish_states:
            self.pub_state = ros.Publisher(self.state_pub_name,
                                           Float64MultiArray,
                                           queue_size=50)
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
            self._publish_states(state)
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
                           pub_actions=True, pub_states=True,
                           save_folder=folder)
            s.run()

        if sys.argv[1] == "cl":
            folder = "data/nn_sim_learning/" + utils.timestamp()
            utils.mkdir(folder)
            s = Simulation(sim_file=sys.argv[2], save_folder=folder,
                           t_sim=1000, t_pretrain=999, t_start_cl=1500,
                           t_stop_cl=1600)
            s.run()

        if sys.argv[1] == "train":
            folder = "data/nn_sim_learning/" + utils.timestamp()
            utils.mkdir(folder)
            s = Simulation(save_folder=folder,
                           t_sim=1000, t_pretrain=999, t_start_cl=1500,
                           t_stop_cl=1600)
            s.run()

        if sys.argv[1] == "cpg":
            s = CPGSimulation()
            s.run()