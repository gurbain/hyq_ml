
import copy
import fileinput
import numpy as np
import os
import pexpect
import psutil
import random
import re
import rospy as ros
import rosgraph
import string
import subprocess
import sys
import time
import threading
import traceback

from gazebo_msgs.srv import ApplyBodyWrench, SetPhysicsProperties
from gazebo_msgs.msg import ODEPhysics
from geometry_msgs.msg import Vector3, Wrench
from rosgraph_msgs.msg import Clock
from dls_msgs.msg import StringDoubleArray
from dwl_msgs.msg import WholeBodyState
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Header
from std_srvs.srv import Empty

import utils

# Remove stupid a,d annoying ROS Launch stderr redirection
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr


class HyQSim(threading.Thread):

    def __init__(self, view=False, rviz=False, init_impedance=None, 
                 verbose=1, inputs=None):

        threading.Thread.__init__(self)

        # Process information
        self.init_impedance = init_impedance
        self.ros_port = "11311"
        self.ros_ps_name = "roscore"
        self.ros_ps = None
        self.sim_ps = None
        self.sim_ps_name = "roslaunch"
        self.sim_package = "dls_supervisor"
        self.sim_node = "operator.launch"
        self.sim_to_kill = ["ros", "gzserver", "gzclient"]
        self.sim_not_kill = ["PlotJuggler", "rosmaster", "rosout", "roscore", "rosbag"]
        self.sim_params = "gazebo:=True osc:=False"
        if view is False:
            self.sim_params += " gui:=False"
        if rviz is False:
            self.sim_params += " rviz:=False"
        self.rcf_config_file = os.path.dirname(os.path.realpath(__file__)) + \
                               "/../config/hyq_sim_options.ini"

        # ROS Topics
        self.clock_sub_name = 'clock'
        self.hyq_state_sub_name = "/hyq/robot_states"
        self.hyq_tgt_action_sub_name = "/hyq/des_rcf_joint_states"
        self.hyq_nn_pub_name = "/hyq/des_nn_joint_states"
        self.hyq_nn_w_pub_name = "/hyq/nn_weight"
        self.hyq_pow_pub_name = "/hyq/power"
        self.sub_clock = None
        self.sub_state = None
        self.sub_power_state = None
        self.sub_tgt_action = None
        self.pub_nn = None
        self.pub_nn_w = None
        self.pub_power = None
        self.srv_noise = None
        self.srv_phys_prop = None

        # Simulation state
        self.sim_time = 0
        self.inputs_len = 0
        if inputs is None:
            inputs = ['bias', 'imu', 'grf', 'joints']
        if inputs is not None:
            if 'bias' in inputs:
                self.inputs_len += 1
            if 'imu' in inputs:
                self.inputs_len += 3
            if 'grf' in inputs:
                self.inputs_len += 4
            if 'joints' in inputs:
                self.inputs_len += 12
        self.hyq_inputs = inputs
        self.hyq_state = None
        self.hyq_tgt_action = None
        self.hyq_full_tgt_action = None
        self.hyq_x = 0
        self.hyq_y = 0
        self.hyq_z = 0
        self.hyq_phi = 0
        self.hyq_theta = 0
        self.hyq_psi = 0
        self.hyq_lf_grf = 0
        self.hyq_rf_grf = 0
        self.hyq_lh_grf = 0
        self.hyq_rh_grf = 0
        self.hyq_fall = False
        self.hyq_power = 0
        self.lpf_power = None
        self.hyq_state_it = 0
        self.hyq_action_it = 0
        self.tc_kp = 5000
        self.tc_kd = 800
        self.process_state_flag = threading.Lock()
        self.process_action_flag = threading.Lock()

        # Class behaviour
        self.t_init = time.time()
        self.controller_started = False
        self.trot_started = False
        self.trunk_cont_stopped = False
        self.sim_started = False
        self.noise_it = 0
        self.next_noise_it = 0
        self.verbose = verbose
        self.daemon = True
        self.stopped = False

    def start_ros(self, verbose):

        try:
            if (self.get_ros_status() == True):
                self.printv("[HyQ Gym] Restarting Roscore")
                self.stop_ros()

            proc = [self.ros_ps_name, "-p", self.ros_port]
            self.ros_ps = pexpect.spawn(" ".join(proc))
            if self.verbose == 2:
                self.ros_ps.logfile_read = sys.stdout

            time.sleep(2)
            os.system("rosparam set /use_sim_time True")
            time.sleep(0.1)
            self.printv("[HyQ Gym] Roscore has started")

        except Exception, e:
            if self.verbose > 0:
                print "Exception encountered during roscore init!" + str(e)
                traceback.print_exc()

    def stop_ros(self):

        # Try to kill correctly first
        if self.ros_ps is not None:
            if self.ros_ps.isalive():
                self.ros_ps.sendcontrol("c")
                self.ros_ps.sendcontrol("d")
                self.ros_ps.sendcontrol("c")
                time.sleep(2)

        # Kill gzclient, gzserver
        tmp = os.popen("ps -Af").read()
        roscore_count = tmp.count('roscore')
        rosmaster_count = tmp.count('rosmaster')

        if rosmaster_count > 0:
            os.system("pkill -9 rosmaster")
            os.system("pkill -9 rosmaster")

        if roscore_count > 0:
            os.system("pkill -9 roscore")
            os.system("pkill -9 rosmaster")

        time.sleep(1)
        while(self.get_ros_status()):
            time.sleep(0.1)

    def get_ros_status(self):

        state_running = False
        ros_status = psutil.STATUS_STOPPED
        for proc in psutil.process_iter():
            if proc.name() == self.ros_ps_name:
                ros_status = proc.status()
        if ros_status == psutil.STATUS_RUNNING or ros_status == psutil.STATUS_SLEEPING:
            state_running = True

        return state_running

    def run(self):

        try:
            # Subsribe to ROS topics
            self.sub_clock = ros.Subscriber(self.clock_sub_name, Clock,
                                            callback=self._reg_sim_time,
                                            queue_size=1)
            self.sub_tgt_action = ros.Subscriber(self.hyq_tgt_action_sub_name,
                                                 JointState,
                                                 callback=self._reg_hyq_tgt_action,
                                                 queue_size=2,
                                                 buff_size=2**20,
                                                 tcp_nodelay=True)
            self.sub_state = ros.Subscriber(self.hyq_state_sub_name,
                                            WholeBodyState,
                                            callback=self._reg_hyq_state,
                                            queue_size=2,
                                            buff_size=2**20,
                                            tcp_nodelay=True)

            self.sub_power_state = ros.Subscriber(self.hyq_state_sub_name,
                                                  WholeBodyState,
                                                  callback=self._reg_hyq_power,
                                                  queue_size=2,
                                                  buff_size=2**20,
                                                  tcp_nodelay=True)

            # Create ROS Publishers
            self.pub_nn = ros.Publisher(self.hyq_nn_pub_name,
                                        JointState,
                                        queue_size=1)
            self.pub_nn_w = ros.Publisher(self.hyq_nn_w_pub_name,
                                          Float32,
                                          queue_size=1)
            self.pub_power = ros.Publisher(self.hyq_pow_pub_name,
                                           Float32,
                                           queue_size=1)

            # Start the physics
            self.start_sim()

            # Wait and start the controller
            self.start_controller()

            # Create ROS proxy
            self.srv_phys_prop = ros.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)


            self.sim_started = True

        except Exception, e:
            if self.verbose > 0:
                print "Exception encountered during simulation!" #+ str(e)

        while not self.stopped:
            time.sleep(0.1)

    def stop(self):

        self.stop_sim()
        self.stopped = True

    def start_sim(self):

        try:
            # Init the config file
            self.set_init_impedances()

            self.t_init = time.time()
            if self.sim_ps is not None:
                if self.sim_ps.isalive():
                    self.printv("[HyQ Gym] Physics is Already Started")
                    return

            proc = [self.sim_ps_name, self.sim_package,
                    self.sim_node, self.sim_params]

            self.printv("[HyQ Gym] Starting Physics")

            self.sim_ps = pexpect.spawn(" ".join(proc))
            if self.verbose == 2:
                self.sim_ps.logfile_read = sys.stdout

        except Exception, e:
            if self.verbose > 0:
                print "Exception encountered during simulation!" + str(e)
                traceback.print_exc()
            if self.sim_ps is not None:
                if self.sim_ps.isalive():
                    self.stop_sim()
    
    def stop_sim(self, clean_stop=False):

        self.printv("[HyQ Gym] Stopping Physics")

        if clean_stop:
            self.sim_ps.sendcontrol("c")
            self.sim_ps.sendcontrol("d")
            self.sim_ps.sendcontrol("c")
            i = 0
            while self.sim_ps.isalive() and i < 300:
                # print 'alive: ' + str(self.sim_ps.isalive()) + " i: " + str(i)
                time.sleep(0.1)
                i += 1

        if self.sim_ps is not None:
            if self.sim_ps.isalive():
                try:
                    for proc in psutil.process_iter():
                        name = " ".join(proc.cmdline())
                        for y in self.sim_to_kill:
                            if y in name:
                                kill_me = True
                                for n in self.sim_not_kill:
                                    if n in name:
                                        kill_me = False
                                if kill_me:
                                    proc.kill()
                except (psutil.ZombieProcess, psutil.AccessDenied, psutil.NoSuchProcess, IOError):
                    pass

    def send_controller_cmd(self, cmd, rsp, timeout=1):

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
                    if self.verbose > 0:
                        print("Command: " + cmd + "\nExpected Response: "+
                              rsp + "\nReceived Response: " + str(self.sim_ps.before))
                pass

        return rsp

    def start_controller(self):

        self.sim_ps.expect("PrepController>>", timeout=25)
        self.send_controller_cmd("trunkCont", "PrepController>>")
        self.send_controller_cmd("changeController", "Indice")

        # RCF Controller
        self.send_controller_cmd("3", "Freeze Base off!", timeout=15)
        self.send_controller_cmd("", "RCFController>>")
        self.send_controller_cmd("kadj", "Kinematic adjustment ON!!!")

        self.printv("[HyQ Gym] RCF Controller is Set")
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
            self.trot_started = True
            return True

        else:
            return False

    def stop_trunk_cont(self):

        self.printv("[HyQ Gym] Stopping Trunk Controller")

        if self.controller_started and not self.trunk_cont_stopped:
            self.send_controller_cmd("trunkCont", "Turning TrunkCont OFF")
            self.printv("[HyQ Gym] Trunk Controller Stopped")
            self.trunk_cont_stopped = True

    def get_tc_z_gain(self):

        rsp = ""
        try:
            self.sim_ps.sendline("whereTrunkGains")
            self.sim_ps.expect("FAKE_ANSWER", timeout=1)
        except pexpect.exceptions.TIMEOUT:
            rsp = str(self.sim_ps.before)
            pass

        self.tc_kp = int(float((rsp.split("Kp Z:"))[1].split("Kd Z")[0].strip()))
        self.tc_kd = int(float((rsp.split("Kd Z:"))[1].split("Robot")[0].strip()))

        return self.tc_kp, self.tc_kd

    def set_tc_z_gain(self, new_kp=5000, new_kd=800):

        if self.controller_started:

            self.send_controller_cmd("changeTrunkControllerGains", "KpTrunkRoll")
            for _ in range(20):
                self.send_controller_cmd("", ":")

            self.send_controller_cmd(str(int(new_kp)), "Are you SURE")
            self.send_controller_cmd("1", "KdTrunkz")
            self.send_controller_cmd(str(int(new_kd)), "Are you SURE")
            self.send_controller_cmd("1", "weight")

            for _ in range(16):
                self.send_controller_cmd("", "")
            self.send_controller_cmd("", "RCFController>>")

    def set_impedances(self, imp):

        # k_vec must be of size 6
        if self.controller_started:
            self.send_controller_cmd("setGains", "haa_joint:")
            # For each leg
            for i in range(3):
                self.send_controller_cmd(str(imp[0]), "Kd")
                self.send_controller_cmd(str(imp[1]), "hfe_joint:")
                self.send_controller_cmd(str(imp[2]), "Kd")
                self.send_controller_cmd(str(imp[3]), "kfe_joint:")
                self.send_controller_cmd(str(imp[4]), "Kd")
                self.send_controller_cmd(str(imp[5]), "haa_joint:")

            self.send_controller_cmd(str(imp[0]), "Kd")
            self.send_controller_cmd(str(imp[1]), "hfe_joint:")
            self.send_controller_cmd(str(imp[2]), "Kd")
            self.send_controller_cmd(str(imp[3]), "kfe_joint:")
            self.send_controller_cmd(str(imp[4]), "Kd")
            self.send_controller_cmd(str(imp[5]), "RCFController>>")

    def set_init_impedances(self):

        f_origin = open(self.rcf_config_file, "r")
        f_new = open(self.rcf_config_file + ".current_sim", "w")

        for line in f_origin:
            if self.init_impedance is not None:
                if line.startswith("KpHAA = "):
                    f_new.write("KpHAA = {0:.3f}".format(float(self.init_impedance[0])) + "\n")
                elif line.startswith("KdHAA = "):
                    f_new.write("KdHAA = {0:.3f}".format(float(self.init_impedance[1])) + "\n")
                elif line.startswith("KpHFE = "):
                    f_new.write("KpHFE = {0:.3f}".format(float(self.init_impedance[2])) + "\n")
                elif line.startswith("KdHFE = "):
                    f_new.write("KdHFE = {0:.3f}".format(float(self.init_impedance[3])) + "\n")
                elif line.startswith("KpKFE = "):
                    f_new.write("KpKFE = {0:.3f}".format(float(self.init_impedance[4])) + "\n")
                elif line.startswith("KdKFE = "):
                    f_new.write("KdKFE = {0:.3f}".format(float(self.init_impedance[5])) + "\n")
                else:
                    f_new.write(line.rstrip() + "\n")
            else:
                f_new.write(line.rstrip() + "\n")

        f_origin.close()
        f_new.close()

    def get_impendances(self):

        imp = dict()
        if self.controller_started:
            try:
                self.sim_ps.sendline("whereGains")
                self.sim_ps.expect("Make fail on purpose", timeout=0.5)
            except pexpect.exceptions.TIMEOUT:
                imp_str = str(self.sim_ps.before)
                imp_str_arr = imp_str.splitlines()
                for a in imp_str_arr:
                    if "Joint: " in a:
                        b = a.replace("=", " ").replace(":", " ").split(" ")
                        imp[b[2]] = [b[6], b[10], b[14]]
                pass

        return imp

    def get_sim_time(self):

        return self.sim_time

    def get_hyq_state(self):

        self.process_state_flag.acquire()
        try:
            inputs = []
            if self.hyq_state is not None:
                inputs = self.hyq_state
        finally:
            self.process_state_flag.release()

        return np.mat(inputs)

    def get_hyq_tgt_action(self):

        self.process_action_flag.acquire()
        try:
            outputs = []
            if self.hyq_tgt_action is not None:
                outputs = self.hyq_tgt_action
        finally:
            self.process_action_flag.release()

        return np.mat(outputs)

    def get_hyq_x_y_z(self):

        return copy.deepcopy(self.hyq_x), \
               copy.deepcopy(self.hyq_y), \
               copy.deepcopy(self.hyq_z)

    def get_hyq_power(self):

        return copy.deepcopy(self.hyq_power)

    def get_hyq_grf(self):

        return copy.deepcopy(self.hyq_lh_grf), \
               copy.deepcopy(self.hyq_lf_grf), \
               copy.deepcopy(self.hyq_rh_grf), \
               copy.deepcopy(self.hyq_rf_grf)

    def get_hyq_phi_theta_psi(self):

        return copy.deepcopy(self.hyq_phi), \
               copy.deepcopy(self.hyq_theta), \
               copy.deepcopy(self.hyq_psi)

    def send_hyq_nn_pred(self, prediction, weight, error=None):

        if type(prediction) is np.ndarray:
            prediction = prediction.tolist()

        if len(prediction) != 12:
                ros.logerr("This method is designed to receive 12 joint" +
                           " position in a specific format!")
                return

        # Create and fill a JointState object
        joints = JointState()
        joints.header = Header()
        joints.header.stamp = ros.Time(self.get_sim_time())

        pos = list(self.hyq_full_tgt_action.position)
        pos[0] = prediction[0]           # LF Hip AA Joint
        pos[1] = prediction[1]           # LF Hip FE Joint
        pos[2] = prediction[2]           # LF Knee FE Joint
        pos[3] = prediction[3]           # RF Hip AA Joint
        pos[4] = prediction[4]           # RF Hip FE Joint
        pos[5] = prediction[5]           # RF Knee FE Joint
        pos[6] = prediction[6]           # LH Hip AA Joint
        pos[7] = prediction[7]           # LH Hip FE Joint
        pos[8] = prediction[8]           # LH Knee FE Joint
        pos[9] = prediction[9]           # RH Hip AA Joint
        pos[10] = prediction[10]         # RH Hip FE Joint
        pos[11] = prediction[11]         # RH Knee FE Joint

        joints.position = tuple(pos)
        joints.velocity = tuple([0]*12)
        joints.effort = tuple([0]*12)

        # Publish
        self.pub_nn.publish(joints)
        self.pub_nn_w.publish(Float32(weight))

    def apply_noise(self, noise_val, noise_it_min, noise_it_max):

        if self.noise_it == 0:
            self.srv_noise = ros.ServiceProxy("/gazebo/apply_body_wrench", ApplyBodyWrench)
            self.next_noise_it = random.randint(noise_it_min, noise_it_max)

        if self.noise_it == self.next_noise_it:
            noise = Wrench()
            noise.force.x = random.uniform(-noise_val, noise_val)
            noise.force.y = random.uniform(-noise_val, noise_val)
            noise.force.z = random.uniform(-noise_val, noise_val)

            try:
                self.srv_noise("hyq::base_link", "", None, noise, ros.Time.now(), ros.Duration.from_sec(1.0))
            except:
                pass

            self.next_noise_it += random.randint(noise_it_min, noise_it_max)

        self.noise_it += 1

    def _reg_sim_time(self, time):

        self.sim_time = time.clock.secs + time.clock.nsecs/1000000000.0

    def _reg_hyq_state(self, msg):

        # Robot State (length = 13)
        inp = []
        if "bias" in self.hyq_inputs:
            inp += [0.5]                              # Bias

        if "grf" in self.hyq_inputs:
            inp += [msg.contacts[0].wrench.force.z]  # LF Foot Z Force
            inp += [msg.contacts[1].wrench.force.z]  # RF Foot Z Force
            inp += [msg.contacts[2].wrench.force.z]  # LH Foot Z Force
            inp += [msg.contacts[3].wrench.force.z]  # RH Foot Z Force

        if "joints" in self.hyq_inputs:
            inp += [msg.joints[0].position]          # LF Hip AA Joint
            inp += [msg.joints[1].position]          # LF Hip FE Joint
            inp += [msg.joints[2].position]          # LF Knee FE Joint
            inp += [msg.joints[3].position]          # RF Hip AA Joint
            inp += [msg.joints[4].position]          # RF Hip FE Joint
            inp += [msg.joints[5].position]          # RF Knee FE Joint
            inp += [msg.joints[6].position]          # LH Hip AA Joint
            inp += [msg.joints[7].position]          # LH Hip FE Joint
            inp += [msg.joints[8].position]          # LH Knee FE Joint
            inp += [msg.joints[9].position]          # RH Hip AA Joint
            inp += [msg.joints[10].position]         # RH Hip FE Joint
            inp += [msg.joints[11].position]         # RH Knee FE Joint

        if "imu" in self.hyq_inputs:
            inp += [msg.base[0].position]            # Base Roll
            inp += [msg.base[1].position]            # Base Yaw
            inp += [msg.base[2].position]            # Base Pitch
            inp += [msg.base[3].position]            # Base X
            inp += [msg.base[4].position]            # Base Y
            inp += [msg.base[5].position]            # Base Z

        self.process_state_flag.acquire()
        try:
            self.hyq_phi = msg.base[0].position
            self.hyq_theta = msg.base[1].position
            self.hyq_psi= msg.base[2].position
            self.hyq_x = msg.base[3].position
            self.hyq_y = msg.base[4].position
            self.hyq_z = msg.base[5].position
            self.hyq_lf_grf = msg.contacts[0].wrench.force.z
            self.hyq_rf_grf = msg.contacts[1].wrench.force.z
            self.hyq_lh_grf = msg.contacts[2].wrench.force.z
            self.hyq_rh_grf = msg.contacts[3].wrench.force.z
            if not self.hyq_fall and self.trot_started:
                if self.hyq_z < 0.35 or abs(self.hyq_phi) > 1.0 or abs(self.hyq_psi) > 1.0:
                    print "[HyQ Gym] The robot has touched the ground because of Z={0:.2f}".format(self.hyq_z) + \
                          " or PHI={0:.2f}".format(self.hyq_phi) + " or PSI={0:.2f}".format(self.hyq_psi)
                    self.hyq_fall = True
            self.hyq_state = inp
            self.hyq_state_it += 1
        finally:
            self.process_state_flag.release()

    def _reg_hyq_power(self, msg):

        power = 0.0
        for i, j in enumerate(msg.joints):
            power += j.velocity * j.effort

        self.hyq_power = power
        self.pub_power.publish(power)

    def _reg_hyq_tgt_action(self, msg):

        # Robot Target Action (length = 12)
        out = [msg.position[0]]           # LF Hip AA Joint
        out += [msg.position[1]]          # LF Hip FE Joint
        out += [msg.position[2]]          # LF Knee FE Joint
        out += [msg.position[3]]          # RF Hip AA Joint
        out += [msg.position[4]]          # RF Hip FE Joint
        out += [msg.position[5]]          # RF Knee FE Joint
        out += [msg.position[6]]          # LH Hip AA Joint
        out += [msg.position[7]]          # LH Hip FE Joint
        out += [msg.position[8]]          # LH Knee FE Joint
        out += [msg.position[9]]          # RH Hip AA Joint
        out += [msg.position[10]]         # RH Hip FE Joint
        out += [msg.position[11]]         # RH Knee FE Joint

        self.process_action_flag.acquire()
        try:
            self.hyq_tgt_action = out
            self.hyq_full_tgt_action = copy.deepcopy(msg)
            self.hyq_action_it += 1
        finally:
            self.process_action_flag.release()

    def set_phys_prop(self, max_update_rate, time_step=0.001, gravity=(0, 0, -9.8)):
        
        gravity = Vector3(*gravity)
        ode_config = ODEPhysics(auto_disable_bodies=False,
                                sor_pgs_precon_iters=0, sor_pgs_iters=50,
                                sor_pgs_w=1.3, sor_pgs_rms_error_tol=0.0,
                                contact_surface_layer=0.001,
                                contact_max_correcting_vel=100.0,
                                cfm=0.0, erp=0.2, max_contacts=20)
        ros.wait_for_service("/gazebo/set_physics_properties")
        try:
            return self.srv_phys_prop(time_step=time_step,
                                      max_update_rate=max_update_rate,
                                      gravity=gravity,
                                      ode_config=ode_config)

            self.printv("[HyQ Gym] Changing Gazebo Freq to: {}".format(max_update_rate))
        except ros.ServiceException as e:
            if self.verbose > 0:
                print("Failed to set physics: %s", e)

    def printv(self, txt):

        if self.verbose >= 1:
            print(txt)
