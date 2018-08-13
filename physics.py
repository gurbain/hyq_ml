
import copy
import numpy as np
import os
import pexpect
import psutil
import random
import rospy as ros
import rosgraph
import sys
import time
import threading
import traceback

from gazebo_msgs.srv import ApplyBodyWrench
from geometry_msgs.msg import Vector3, Wrench
from rosgraph_msgs.msg import Clock
from dls_msgs.msg import StringDoubleArray
from dwl_msgs.msg import WholeBodyState, WholeBodyTrajectory, JointState, ContactState, BaseState
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Header
from std_srvs.srv import Empty


stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import utils
sys.stderr = stderr


class HyQSim(threading.Thread):

    def __init__(self, view=False, verbose=2, publish_error=False):

        threading.Thread.__init__(self)

        # Process information
        self.sim_ps = None
        self.sim_ps_name = "roslaunch"
        self.sim_package = "dls_supervisor"
        self.sim_node = "hyq.launch"
        self.sim_to_kill = "ros"
        self.sim_not_kill = ""
        self.sim_params = "gazebo:=True state_est:=GroundTruth"
        if view == False:
            self.sim_params += " gui:=False"

        # ROS Services
        self.reset_sim_service = '/gazebo/reset_simulation'
        self.pause_sim_service = '/gazebo/pause_physics'
        self.unpause_sim_service = '/gazebo/unpause_physics'
        self.reset_sim_proxy = None
        self.pause_sim_proxy = None
        self.unpause_sim_proxy = None

        # ROS Topics
        self.clock_sub_name = 'clock'
        self.hyq_state_sub_name = "/hyq/robot_states"
        self.hyq_action_sub_name = "/hyq/des_rcf_joint_states"
        self.hyq_debug_sub_name = "/hyq/debug"
        self.hyq_plan_pub_name = "/hyq/plan"
        self.hyq_nn_pub_name = "/hyq/des_nn_joint_states"
        self.hyq_nn_w_pub_name = "/hyq/nn_weight"
        self.hyq_js_err_name = "/hyq/nn_rcf_js_error"
        self.hyq_haa_err_name = "/hyq/nn_rcf_haa_pos_error"
        self.hyq_hfe_err_name = "/hyq/nn_rcf_hfe_pos_error"
        self.hyq_kfe_err_name = "/hyq/nn_rcf_kfe_pos_error"
        self.hyq_haad_err_name = "/hyq/nn_rcf_haa_vel_error"
        self.hyq_hfed_err_name = "/hyq/nn_rcf_hfe_vel_error"
        self.hyq_kfed_err_name = "/hyq/nn_rcf_kfe_vel_error"
        self.hyq_tot_err_name = "/hyq/nn_rcf_tot_error"
        self.sub_clock = None
        self.sub_state = None
        self.sub_action = None
        self.sub_debug = None
        self.pub_plan = None
        self.pub_nn = None
        self.pub_js_err = None
        self.pub_haa_err = None
        self.pub_hfe_err = None
        self.pub_kfe_err = None
        self.pub_haad_err = None
        self.pub_hfed_err = None
        self.pub_kfed_err = None
        self.pub_tot_err = None

        # Simulation state
        self.sim_time = 0
        self.hyq_state = dict()
        self.hyq_action = dict()
        self.hyq_init_wbs = None
        self.hyq_wbs = None
        self.hyq_state_it = 0
        self.hyq_traj = []
        self.hyq_traj_it = 0
        self.process_state_flag = threading.Lock()
        self.process_action_flag = threading.Lock()

        # Class behaviour
        self.publish_error = publish_error
        self.error_it = 0
        self.error_pos = np.array([])
        self.error_vel = np.array([])
        self.noise_it = 0
        self.verbose = verbose
        self.daemon = True

    def run(self):

        try:
            # Start the roslaunch file
            self.start_sim()

            # Wait and start the controller
            self.start_controller()

            # Get ROS services
            self.reset_sim_proxy = ros.ServiceProxy(self.reset_sim_service, Empty)
            self.pause_sim_proxy = ros.ServiceProxy(self.pause_sim_service, Empty)
            self.unpause_sim_proxy = ros.ServiceProxy(self.unpause_sim_service, Empty)

            # Subsribe to ROS topics
            self.sub_clock = ros.Subscriber(self.clock_sub_name, Clock,
                                            callback=self._reg_sim_time,
                                            queue_size=1)
            self.sub_action = ros.Subscriber(self.hyq_action_sub_name,
                                            JointState,
                                            callback=self._reg_hyq_action,
                                            queue_size=2,
                                            buff_size=2**20,
                                            tcp_nodelay=True)
            self.sub_state = ros.Subscriber(self.hyq_state_sub_name,
                                            WholeBodyState,
                                            callback=self._reg_hyq_state,
                                            queue_size=2,
                                            buff_size=2**20,
                                            tcp_nodelay=True)
            self.sub_debug = ros.Subscriber(self.hyq_debug_sub_name,
                                            StringDoubleArray,
                                            callback=self._reg_hyq_debug,
                                            queue_size=1)

            # Create ROS Publishers
            self.pub_plan = ros.Publisher(self.hyq_plan_pub_name,
                                          WholeBodyTrajectory,
                                          queue_size=1)
            self.pub_nn = ros.Publisher(self.hyq_nn_pub_name,
                                        JointState,
                                        queue_size=1)
            self.pub_nn_w = ros.Publisher(self.hyq_nn_w_pub_name,
                                          Float32,
                                          queue_size=1)

            if self.publish_error:

                self.pub_js_err = ros.Publisher(self.hyq_js_err_name,
                                              JointState,
                                              queue_size=1)
                self.pub_tot_err = ros.Publisher(self.hyq_tot_err_name,
                                              Float32,
                                              queue_size=1)
                self.pub_haa_err = ros.Publisher(self.hyq_haa_err_name,
                                              Float32,
                                              queue_size=1)
                self.pub_hfe_err = ros.Publisher(self.hyq_hfe_err_name,
                                              Float32,
                                              queue_size=1)
                self.pub_kfe_err = ros.Publisher(self.hyq_kfe_err_name,
                                              Float32,
                                              queue_size=1)
                self.pub_haad_err = ros.Publisher(self.hyq_haad_err_name,
                                              Float32,
                                              queue_size=1)
                self.pub_hfed_err = ros.Publisher(self.hyq_hfed_err_name,
                                              Float32,
                                              queue_size=1)
                self.pub_kfed_err = ros.Publisher(self.hyq_kfed_err_name,
                                              Float32,
                                              queue_size=1)


        except Exception, e:
            if self.verbose > 0:
                print "Exception encountered during simulation: " + str(e)
                traceback.print_exc()
            self.stop_sim()
            raise e

    def stop(self):

        self.stop_sim()

    def start_sim(self):

        self.sim_duration = 0
        if self.sim_ps is not None:
            if self.sim_ps.isalive():
                self.printv("\n ===== Physics is Already Started =====\n")
                return

        proc = [self.sim_ps_name, self.sim_package,
                self.sim_node, self.sim_params]

        self.printv("\n ===== Starting Physics =====\n")

        self.sim_ps = pexpect.spawn(" ".join(proc))
        if self.verbose == 2:
            self.sim_ps.logfile_read = sys.stdout

    def stop_sim(self):

        self.printv("\n\n ===== Stopping Physics =====\n")

        self.sim_ps.sendcontrol('c')

        # Kill all other ros processes
        for proc in psutil.process_iter():
            name = " ".join(proc.cmdline())
            if self.sim_to_kill in name and \
               not self.sim_not_kill in name:
                    proc.kill()

    def register_node(self):

        # Create this python ROS node
        # This method cannot be called from the class itself
        # as it has to run on main thread
        with utils.Capturing() as output:
            ros.init_node('physics', anonymous=True)
            print output

    def reset_sim(self):

        ros.wait_for_service(self.reset_sim_service)
        try:
            self.reset_sim_proxy()
        except ros.ServiceException as e:
            print("Reset simulation service call failed with error" + str(e))

    def pause_sim(self):

        ros.wait_for_service(self.pause_sim_service)
        try:
            self.pause_sim_proxy()
        except ros.ServiceException as e:
            print("Pause simulation service call failed with error" + str(e))

    def unpause_sim(self):

        ros.wait_for_service(self.unpause_sim_service)
        try:
            self.unpause_sim_proxy()
        except ros.ServiceException as e:
            print("Unpause simulation service call failed with error" + str(e))

    def step_sim(self, timestep):

        #ros.wait_for_service(self.step_sim_service)
        try:
            self.step_sim_proxy(timestep)
        except ros.ServiceException as e:
            print("Simulation step service call failed with error" + str(e))

    def send_controller_cmd(self, cmd, rsp, timeout=1):

        # Set the Trunk Controller
        i = 0
        while 1:
            if i > 5:
                return -1
            try:
                self.sim_ps.sendline(cmd)
                self.sim_ps.expect(".*" + rsp + ".*", timeout=timeout)
                break
            except pexpect.exceptions.TIMEOUT:
                if self.verbose > 0:
                    print "Command timed out; retrying!"
                    if self.verbose > 1:
                        print(str(self.sim_ps.before))
                pass

    def start_controller(self):

        self.sim_ps.expect("PrepController>>", timeout=25)
        self.send_controller_cmd("trunkCont", "PrepController>>")
        self.send_controller_cmd("changeController", "Indice")

        # Plan Controller
        # self.send_controller_cmd("4", "VMForceOptimizationController>>")
        # self.send_controller_cmd("executePlan", "Executing Plan On")
        # self.send_controller_cmd("changePDgains", "P gain")
        # self.send_controller_cmd("500", "D gain")
        # self.send_controller_cmd("6", "VMForceOptimizationController>>")
        # self.printv("\n\n ===== Plan Controller is Set =====\n")

        # RCF Controller
        self.send_controller_cmd("3", "Freeze Base off!", timeout=15)
        self.send_controller_cmd("", "RCFController>>")
        self.send_controller_cmd("kadj", "Kinematic adjustment ON!!!")
        self.send_controller_cmd("prec", "Push Recovery ON!!!")
        self.send_controller_cmd("narrowStance",
                                 "Narrowing stance posture!!!")
        self.send_controller_cmd("narrowStance",
                                 "Narrowing stance posture!!!")

        self.printv("\n ===== RCF Controller is Set =====\n")

        self.controller_started = True

    def start_rcf_trot(self):

        if self.controller_started:
            self.send_controller_cmd("stw", "WALKING TROT Started!!!")
            self.send_controller_cmd("ctp", "Stance Hs")
            self.send_controller_cmd("", "Comp. touch-down errors")
            self.send_controller_cmd("", "Step Height ")
            self.send_controller_cmd("", "Forward Velocity")
            self.send_controller_cmd("0.25", "Step Frequency")
            for i in range(8):
                self.send_controller_cmd("", ":")
            self.send_controller_cmd("", "RCFController>>")
        else:
            time.sleep(1)
            self.start_rcf_trot()

    def get_sim_time(self):

        return self.sim_time

    def get_hyq_state(self):

        self.process_state_flag.acquire()
        try:
            if set(["base", "joint", "stance"]).issubset(
                                        self.hyq_state.keys()):
                mat = np.mat(self.hyq_state["base"] + \
                             self.hyq_state["joint"] + \
                             self.hyq_state["stance"])
            else:
                mat = np.mat([])
        finally:
            self.process_state_flag.release()

        return mat

    def get_hyq_action(self):

        if set(["pos", "vel"]).issubset(self.hyq_action.keys()):
            return np.mat(self.hyq_action["pos"] + self.hyq_action["vel"])
        else:
            return np.mat([])

    def get_hyq_joints(self):

        if self.hyq_wbs is not None:

            p = [r.position for r in \
                 copy.deepcopy(self.hyq_wbs.joints)]
            v = [r.velocity for r in \
                 copy.deepcopy(self.hyq_wbs.joints)]
            return p + v

    def set_hyq_action(self, action):

        if type(action) is np.ndarray:
            action = action.tolist()

        if len(action) != 24:
                ros.logerr("This method is designed to receive 12 joint" + \
                          " position and velocity in a specific format!")
                return

        if self.hyq_wbs is not None:

            # Get a copy of the current state
            new_point = self.hyq_wbs

            # Add timestamp
            new_point.header = Header()
            new_point.header.stamp = ros.Time(self.get_sim_time())

            # Fill the new joint states with the vector in a given order
            for i, j in enumerate(new_point.joints):
                j.position = action[i]
                j.velocity = action[12+i]
                j.effort = 0.0
                j.acceleration = 0.0

            # Append new trajectory point
            self.hyq_traj.append(new_point)

    def send_hyq_traj(self):

        if len(self.hyq_traj) > 0 and self.hyq_wbs is not None:

            # Get actual whole body state and copy it
            curr = self.hyq_init_wbs
            traj = WholeBodyTrajectory()

            # Add timestamp
            traj.header = Header()
            traj.header.stamp = ros.Time(self.get_sim_time())

            # Write it in the actual value
            traj.actual = curr

            # Transform the actual trajectory with the changed values
            traj.trajectory = self.hyq_traj

            # Publish the message
            self.pub_plan.publish(traj)
            self.hyq_traj_it += 1

        # Flush the trajectory EVEN if not sent
        self.hyq_traj = []

    def send_hyq_nn_pred(self, prediction, weight, error=None):

        if type(prediction) is np.ndarray:
            prediction = prediction.tolist()

        if len(prediction) != 24:
                ros.logerr("This method is designed to receive 12 joint" + \
                          " position and velocity in a specific format!")
                return

        # Create and fill a JointState object
        joints = JointState()
        joints.header = Header()
        joints.header.stamp = ros.Time(self.get_sim_time())
        joints.position = prediction[0:12]
        joints.velocity = prediction[12:24]
        joints.effort = [0.0] * 12

        # Publish
        self.pub_nn.publish(joints)
        self.pub_nn_w.publish(Float32(weight))


        # Create and fill a JointState object for the errors
        if self.publish_error and error is not None:
            self.publish_errs(error)

    def apply_noise(self):


        if self.noise_it == 0:
            self.srv_noise = ros.ServiceProxy("/gazebo/apply_body_wrench", ApplyBodyWrench)
            self.next_noise_it = random.randint(5000, 15000)

        if self.noise_it == self.next_noise_it :
            noise = Wrench()
            noise.force.x = random.uniform(-50, 50)
            noise.force.y = random.uniform(-50, 50)
            noise.force.z = random.uniform(-50, 50)

            #print "Bim", noise, self.noise_it
            try:
                self.srv_noise("hyq::base_link", "", None, noise,
                               ros.Time.now(), ros.Duration.from_sec(1.0))
            except:
                pass

            self.next_noise_it += random.randint(2000, 15000)

        self.noise_it += 1

    def publish_errs(self, error):

        error_pos = error[0:12]
        error_vel = error[12:24]

        # Always publish the joint error
        joints_err = JointState()
        joints_err.header = Header()
        joints_err.header.stamp = ros.Time(self.get_sim_time())
        joints_err.position = error_pos
        joints_err.velocity = error_vel
        joints_err.effort = [0.0] * 12
        self.pub_js_err.publish(joints_err)

        # Average and publish error summaries
        if self.error_it % 50 == 49:
            haa_err = np.average(error_pos[0::3])
            hfe_err = np.average(error_pos[1::3])
            kfe_err = np.average(error_pos[2::3])
            haad_err = np.average(error_vel[0::3])
            hfed_err = np.average(error_vel[1::3])
            kfed_err = np.average(error_vel[2::3])
            tot_err = np.average(error_pos) + np.average(error_vel)
            self.pub_haa_err.publish(Float32(haa_err))
            self.pub_hfe_err.publish(Float32(hfe_err))
            self.pub_kfe_err.publish(Float32(kfe_err))
            self.pub_haad_err.publish(Float32(haad_err))
            self.pub_hfed_err.publish(Float32(hfed_err))
            self.pub_kfed_err.publish(Float32(kfed_err))
            self.pub_tot_err.publish(Float32(tot_err))
            self.error_pos = error_pos
            self.error_vel = error_vel
        else:
            self.error_pos = np.concatenate([self.error_pos, error_pos])
            self.error_vel = np.concatenate([self.error_vel, error_vel])

        self.error_it += 1

    def init_hyq_pose(self):

        self.set_hyq_action([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.send_hyq_traj()

    def _reg_sim_time(self, time):

        self.sim_time = time.clock.secs + time.clock.nsecs/1000000000.0

    def _reg_hyq_state(self, msg):

        if self.hyq_state_it == 0:
            self.hyq_init_wbs = copy.deepcopy(msg)

        self.hyq_wbs = copy.deepcopy(msg)

        self.process_state_flag.acquire()
        try:
            self.hyq_state["base"] = [b.velocity for b in msg.base]
            self.hyq_state["joint"] = [b.position for b in msg.joints]
            self.hyq_state["joint"] += [b.velocity for b in msg.joints]
            self.hyq_state["joint"] += [b.effort for b in msg.joints]

        finally:
            self.process_state_flag.release()

        self.hyq_state_it += 1

    def _reg_hyq_action(self, msg):

        pos = []
        for i in range(len(msg.position)):
            pos.append(msg.position[i])
        vel = []
        for i in range(len(msg.velocity)):
            vel.append(msg.velocity[i])

        self.hyq_action["pos"] = pos
        self.hyq_action["vel"] = vel

    def _reg_hyq_debug(self, msg):

        stance = []
        for i, m in enumerate(msg.name):
            if m in ["stanceLF", "stanceLH",
                     "stanceRF", "stanceRH"]:
                stance += [msg.data[i]]

        self.hyq_state["stance"] = stance

    def printv(self, txt):

        if self.verbose >= 1:
            print(txt)


if __name__ == '__main__':

    # Create and start simulation
    p = HyQSim(view=True)
    p.start()
    p.register_node()

    if len(sys.argv) > 1:

        if sys.argv[1] == "sine":
            while not ros.is_shutdown():
                t = p.get_sim_time()
                pos = 15 * np.pi / 180 * np.sin(2 * np.pi * 2 * t)
                vel = 15 * np.pi / 180 * np.cos(2 * np.pi * 2 * t)
                p.set_hyq_action([0, 0, pos, 0, 0, pos, 0, 0, pos, 0, 0, pos,
                                  0, 0, vel, 0, 0, vel, 0, 0, vel, 0, 0, vel])
                p.send_hyq_traj()

        if sys.argv[1] == "rcf":
            trot_flag = False
            for i in range(120):
                print("Time: " + str(i) + "s and Sim time: " + \
                      str(p.get_sim_time()) + "s and state (len= " + \
                      str(np.array(p.get_hyq_state()).shape) + "): " + \
                      str(np.array(p.get_hyq_state())))
                if trot_flag == False:
                    if p.get_sim_time() > 1:
                        p.start_rcf_trot()
                        trot_flag = True
                if ros.is_shutdown():
                    p.stop()
                    exit(-1)
                time.sleep(1)

    else:
        for i in range(120):
            print("Time: " + str(i) + "s and Sim time: " + \
                  str(p.get_sim_time()) + "s and state (len= " + \
                  str(np.array(p.get_hyq_state()).shape) + "): " + \
                  str(np.array(p.get_hyq_state())))
            if ros.is_shutdown():
                p.stop()
                exit(-1)
            time.sleep(1)

    # Stop simulation
    p.stop()