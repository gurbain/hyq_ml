import numpy as np
import rospy as ros
import rosbag
import sys
import time

from std_msgs.msg import Float64MultiArray, MultiArrayLayout

import physics
import network

class Simulation():

    def __init__(self, nn_folder, verbose=0, view=False, pub_actions=True,
                 pub_states=True):

        self.t_sim  = 180
        self.t_start_cl = 15
        self.t_stop_cl = 160
        self.t = 0
        self.it = 0
        self.t_cl = self.t_stop_cl - self.t_start_cl

        self.view = view
        self.verbose = verbose
        self.nn_folder = nn_folder
        self.publish_actions = pub_actions
        self.publish_states = pub_states

        self.state_pub_name = "/sim_debug/states"
        self.pred_pub_name = "/sim_debug/prediction"
        self.target_pub_name = "/sim_debug/target"
        self.mix_pub_name = "/sim_debug/mix_cl"

    def start(self):

        # Create and start the handle thread to the physics simulation
        self.physics = physics.HyQSim(view=self.view, verbose=self.verbose)
        self.physics.start()
        self.physics.register_node()

        # Create and load the trained network
        self.network = network.FeedForwardNN(verbose=self.verbose)
        self.network.load(self.nn_folder, load_all=True)
        self.network.verbose = self.verbose

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

        # Retrieve interpolation functions for the target and del the rest
        self.target_fct = self.network.get_fct(self.network.y_t,
                                               self.network.y_val)
        del self.network.y_t, self.network.y_val, \
            self.network.x_t, self.network.x_val, \
            self.network.x2_t, self.network.x2_val

    def get_actions(self, state):

        # Load target action
        target = self.network.interpolate(self.target_fct, self.t)

        # Predict network action
        if state.shape[1] == 46:
            predicted = self.network.predict(state)
        else:
            predicted = None

        return predicted, target

    def step(self):

        # Get time
        self.t = self.physics.get_sim_time()

        # Get last state
        state = self.physics.get_hyq_state()

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

            # Send action
            self.physics.set_hyq_action(action)
            self.physics.send_hyq_traj()

            # Publish data on ROS topic to debug
            self._publish_states(state)
            self._publish_actions(predicted, target, action)

    def run(self):

        self.start()

        self.printv("\n\n ===== Running the Simulation Loop =====\n")

        last_it = 0
        last_t = 0
        while not ros.is_shutdown() and self.t < self.t_sim:
            self.step()

            if self.it % 1000 == 0 and self.t > 0 and self.verbose > 1:
                f = (self.it - last_it) / (self.t - last_t)
                print("Iteration: " + str(self.it) + \
                     "\tSim Time: " + "{:.2f}".format(self.t) + " s" + \
                     "\tNN Freq: " + "{:.2f}".format(f) + " Hz")
                last_it = self.it
                last_t = self.t

            self.it += 1

        self.stop()

    def stop(self):

        self.physics.stop()
        del self.network, self.physics

    def _publish_actions(self, pred, target, action):

        if self.publish_actions:
            if pred is not None:
                pred_array = Float64MultiArray(data=pred.tolist())
                self.pub_pred.publish(pred_array)
            if target is not None:
                target_array = Float64MultiArray(data=target.tolist())
                self.pub_target.publish(target_array)
            if action is not None:
                mix_array = Float64MultiArray(data=action.tolist())
                self.pub_mix.publish(mix_array)

    def _publish_states(self, state):

        if self.publish_states:
            if state is not None:
                state_array = Float64MultiArray(data=state.tolist()[0])
                self.pub_state.publish(state_array)

    def printv(self, txt):

        if self.verbose >= 1:
            print(txt)

if __name__ == '__main__':

    if len(sys.argv) > 1:
        s = Simulation(nn_folder=sys.argv[1])
        s.run()

