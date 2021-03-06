
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
import sys
import time
import tf
import threading
import traceback

from gazebo_msgs.srv import ApplyBodyWrench
from geometry_msgs.msg import Vector3, Wrench, Quaternion, Pose, Point, Polygon, PolygonStamped
from rosgraph_msgs.msg import Clock
from dls_msgs.msg import StringDoubleArray
from dwl_msgs.msg import WholeBodyState
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Header, Float32MultiArray, ColorRGBA
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker

import processing
import utils

# Remove stupid a,d annoying ROS Launch stderr redirection
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr


class HyQSim(threading.Thread):

    def __init__(self, view=False, rviz=False, rt=True, remote=False, fast=False,
                 init_impedance=None, verbose=1, publish_error=False,
                 publish_markers=True, inputs=None):

        threading.Thread.__init__(self)

        # Process information
        self.init_impedance = init_impedance
        self.remote = remote
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
        if fast:
            self.sim_params += " world_name:=empty_fast.world"
        self.rcf_config_file = os.path.dirname(os.path.realpath(__file__)) + \
                               "/../config/hyq_sim_options.ini"

        # ROS Topics
        self.publish_markers = publish_markers
        self.clock_sub_name = 'clock'
        self.hyq_state_sub_name = "/hyq/robot_states"
        self.hyq_tgt_action_sub_name = "/hyq/des_rcf_joint_states"
        self.hyq_nn_pub_name = "/hyq/des_nn_joint_states"
        self.hyq_nn_w_pub_name = "/hyq/nn_weight"
        self.hyq_vel_pub_name = "/hyq/vel"
        self.hyq_pow_pub_name = "/hyq/power"
        self.hyq_lh_mark_pub_name = "/hyq/lh_mark"
        self.hyq_rh_mark_pub_name = "/hyq/rh_mark"
        self.hyq_lf_mark_pub_name = "/hyq/lf_mark"
        self.hyq_rf_mark_pub_name = "/hyq/rf_mark"
        self.hyq_base_mark_pub_name = "/hyq/base_mark"
        self.hyq_phase_poly_pub_name = "/hyq/phase_poly"
        self.hyq_grf_th_pub_name = "/hyq/grf_th"
        self.hyq_js_err_name = "/hyq/nn_rcf_js_error"
        self.hyq_haa_err_name = "/hyq/nn_rcf_haa_pos_error"
        self.hyq_hfe_err_name = "/hyq/nn_rcf_hfe_pos_error"
        self.hyq_kfe_err_name = "/hyq/nn_rcf_kfe_pos_error"
        self.hyq_haad_err_name = "/hyq/nn_rcf_haa_vel_error"
        self.hyq_hfed_err_name = "/hyq/nn_rcf_hfe_vel_error"
        self.hyq_kfed_err_name = "/hyq/nn_rcf_kfe_vel_error"
        self.hyq_tot_err_name = "/hyq/nn_rcf_tot_error"
        self.hyq_foot_pos_1 = [0, 0, 0]
        self.hyq_foot_pos_2 = [0, 0, 0]
        self.sub_clock = None
        self.sub_state = None
        self.sub_power_state = None
        self.sub_tgt_action = None
        self.list_tf = None
        self.pub_nn = None
        self.pub_nn_w = None
        self.pub_power = None
        self.pub_js_err = None
        self.pub_haa_err = None
        self.pub_hfe_err = None
        self.pub_kfe_err = None
        self.pub_haad_err = None
        self.pub_hfed_err = None
        self.pub_kfed_err = None
        self.pub_tot_err = None
        self.pub_grf_th = None
        self.pub_lh_mark = None
        self.pub_lf_mark = None
        self.pub_rh_mark = None
        self.pub_rf_mark = None
        self.pub_base_mark = None
        self.pub_mark_rate = None
        self.pub_poly = None
        self.srv_noise = None

        # Simulation state
        self.sim_time = 0
        self.sim_time_init = 0
        self.hyq_inputs = inputs
        self.inputs_len = 0
        if 'bias' in inputs:
            self.inputs_len += 1
        if 'imu' in inputs:
            self.inputs_len += 3
        if 'grf' in inputs:
            self.inputs_len += 4
        if 'grf_th' in inputs:
            self.inputs_len += 4
            self.th_n = 10
            self.th_val = 60
            self.c1_buff = [0] * self.th_n
            self.c2_buff = [0] * self.th_n
            self.c3_buff = [0] * self.th_n
            self.c4_buff = [0] * self.th_n
        if 'joints' in inputs:
            self.inputs_len += 8
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
        self.rt = rt
        self.t_init = time.time()
        self.controller_started = False
        self.trot_started = False
        self.trunk_cont_stopped = False
        self.sim_started = False
        self.publish_error = publish_error
        self.error_it = 0
        self.error_pos = np.array([])
        self.noise_it = 0
        self.next_noise_it = 0
        self.verbose = verbose
        self.daemon = True
        self.stopped = False
        self.t_stopped = False

        self.vel_wf = processing.WindowFilter(ord=20)
        self.vel_lpf = processing.LowPassFilter(fc=3, ts=0.004, ord=20)

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

            self.pub_vel = ros.Publisher(self.hyq_vel_pub_name,
                                         Float32MultiArray,
                                         queue_size=1)
            if self.publish_markers:
                self.pub_mark_thread = threading.Thread(target=self.pub_mark)
                self.pub_mark_thread.start()
            if 'grf_th' in self.hyq_inputs:
                self.pub_grf_th = ros.Publisher(self.hyq_grf_th_pub_name,
                                                Float32MultiArray,
                                                queue_size=1)

            # Create TF Listener
            self.list_tf = tf.TransformListener()

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

            # Initialize time
            if not self.rt:
                if self.sim_time != 0:
                    self.printv(" ===== Reseting Init Sim Time to " + str(self.sim_time) + " =====\n")
                    self.sim_time_init = copy.copy(self.sim_time)

            # Start the physics
            self.start_sim()

            # Wait and start the controller
            self.start_controller()

            self.sim_started = True

        except Exception, e:
            if self.verbose > 0:
                print "Exception encountered during simulation!" + str(e)
                traceback.print_exc()
            if self.sim_ps is not None:
                if self.sim_ps.isalive():
                    self.stop_sim()

        i = 0
        while not self.stopped:
            if i%100 == 99:
                self.draw_plane(add=False)
            time.sleep(0.1)
            i += 1

    def stop(self):

        self.t_stopped = True
        self.stop_sim()
        self.stopped = True

    def start_sim(self):

        # If the physics should not on this computer, just do not do anything
        if self.remote:
            self.printv(" ===== Physics should run on remote computer =====\n")
            return

        try:
            # Init the config file
            self.set_init_impedances()

            self.t_init = time.time()
            if self.sim_ps is not None:
                if self.sim_ps.isalive():
                    self.printv(" ===== Physics is Already Started =====\n")
                    return

            proc = [self.sim_ps_name, self.sim_package,
                    self.sim_node, self.sim_params]

            self.printv(" ===== Starting Physics =====\n")

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

        if self.remote:
            return

        self.printv(" ===== Stopping Physics =====")

        if self.publish_markers:
            self.pub_mark_thread.join()

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

    def register_node(self):

        # Create this python ROS node
        # This method cannot be called from the class itself
        # as it has to run on main thread
        with utils.Capturing() as output:
            ros.init_node("physics", anonymous=True, disable_signals=True)
            print output

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

        if self.remote:
            return

        self.sim_ps.expect("PrepController>>", timeout=25)
        self.send_controller_cmd("trunkCont", "PrepController>>")
        self.send_controller_cmd("changeController", "Indice")

        # RCF Controller
        self.send_controller_cmd("3", "Freeze Base off!", timeout=15)
        self.send_controller_cmd("", "RCFController>>")
        self.send_controller_cmd("kadj", "Kinematic adjustment ON!!!")

        self.printv(" ===== RCF Controller is Set =====\n")
        self.controller_started = True

    def start_rcf_trot(self):

        if self.remote:
            return True

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

        self.printv(" ===== Stopping Trunk Controller =====\n")

        if self.controller_started and not self.trunk_cont_stopped:
            self.send_controller_cmd("trunkCont", "Turning TrunkCont OFF")
            self.printv(" ===== Trunk Controller Stopped =====\n")
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

        if not self.rt:
            return self.sim_time - self.sim_time_init
        else:
            return time.time() - self.t_init

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

    def get_hyq_foot_pos(self):

        try:
            (t_trunk_foot, r) = self.list_tf.lookupTransform('/trunk', '/lh_foot', ros.Time(0))
            (t_foot, r) = self.list_tf.lookupTransform('/world', '/lh_foot', ros.Time(0))
            (t_trunk, r) = self.list_tf.lookupTransform('/world', '/trunk', ros.Time(0))

            self.hyq_foot_pos_1 = [t_foot[0] - t_trunk[0], t_foot[1] - t_trunk[1], t_foot[2]]
            self.hyq_foot_pos_2 = [t_trunk_foot[0], t_trunk_foot[1], t_trunk_foot[2]]

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

        return copy.deepcopy(self.hyq_foot_pos_1), copy.deepcopy(self.hyq_foot_pos_2)

    def reset_hyq_nn_weight(self):

        for i in range(400):
            self.pub_nn_w.publish(Float32(0))
            time.sleep(0.01)

    def send_hyq_nn_pred(self, prediction, weight, error=None):

        if type(prediction) is np.ndarray:
            prediction = prediction.tolist()

        if len(prediction) != 16:
                ros.logerr("This method is designed to receive 16 joint" +
                           " position and velocity in a specific format!")
                return

        # Create and fill a JointState object
        joints = JointState()
        joints.header = Header()
        if self.rt:
            joints.header.stamp = ros.get_rostime()
        else:
            joints.header.stamp = ros.Time(self.get_sim_time())

        pos = list(self.hyq_full_tgt_action.position)
        pos[1] = prediction[0]           # LF Hip FE Joint
        pos[2] = prediction[1]           # LF Knee FE Joint
        pos[4] = prediction[2]           # RF Hip FE Joint
        pos[5] = prediction[3]           # RF Knee FE Joint
        pos[7] = prediction[4]           # LH Hip FE Joint
        pos[8] = prediction[5]           # LH Knee FE Joint
        pos[10] = prediction[6]          # RH Hip FE Joint
        pos[11] = prediction[7]          # RH Knee FE Joint

        joints.position = tuple(pos)
        #print self.hyq_full_tgt_action.position, joints.position
        #joints.velocity = self.hyq_full_tgt_action.velocity 
        vel = list(self.hyq_full_tgt_action.velocity)
        vel[1] = prediction[8]           # LF Hip FE Joint
        vel[2] = prediction[9]           # LF Knee FE Joint
        vel[4] = prediction[10]           # RF Hip FE Joint
        vel[5] = prediction[11]           # RF Knee FE Joint
        vel[7] = prediction[12]           # LH Hip FE Joint
        vel[8] = prediction[13]           # LH Knee FE Joint
        vel[10] = prediction[14]          # RH Hip FE Joint
        vel[11] = prediction[15]          # RH Knee FE Joint
        joints.velocity = vel

        joints.effort = self.hyq_full_tgt_action.effort

        # joints.position = tuple(pos)
        # t = self.get_sim_time()
        # dt = (t - self.prev_pos_t)
        # #tuple([0] * 12)
        # self.pub_vel.publish(data=self.vel_lpf.transform(np.mat(self.vel_wf.transform(np.mat([(a - b) / dt if dt != 0 else 0 for a, b in zip(pos, self.prev_pos)])))).tolist()[0])
        #self.prev_pos = pos
        #self.prev_pos_t = t

        # Publish
        self.pub_nn.publish(joints)
        self.pub_nn_w.publish(Float32(weight))

        # Create and fill a JointState object for the errors
        if self.publish_error and error is not None:
            self.publish_errs(error)

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

    def publish_errs(self, error):

        # Always publish the joint error
        joints_err = JointState()
        joints_err.header = Header()
        joints_err.header.stamp = ros.Time(self.get_sim_time())
        joints_err.position = [0.0] * 12
        joints_err.velocity = [0.0] * 12
        joints_err.effort = [0.0] * 12

        joints_err.position[1] = error[0]           # LF Hip FE Joint
        joints_err.position[2] = error[1]           # LF Knee FE Joint
        joints_err.position[4] = error[2]           # RF Hip FE Joint
        joints_err.position[5] = error[3]           # RF Knee FE Joint
        joints_err.position[7] = error[4]           # LH Hip FE Joint
        joints_err.position[8] = error[5]           # LH Knee FE Joint
        joints_err.position[10] = error[6]          # RH Hip FE Joint
        joints_err.position[11] = error[7]          # RH Knee FE Joint

        self.pub_js_err.publish(joints_err)

        # Average and publish error summaries
        if self.error_it % 50 == 49:
            hfe_err = np.average(error[0::2])
            kfe_err = np.average(error[1::2])
            tot_err = np.average(error)
            self.pub_haa_err.publish(Float32(0))
            self.pub_hfe_err.publish(Float32(hfe_err))
            self.pub_kfe_err.publish(Float32(kfe_err))
            self.pub_haad_err.publish(Float32(0))
            self.pub_hfed_err.publish(Float32(0))
            self.pub_kfed_err.publish(Float32(0))
            self.pub_tot_err.publish(Float32(tot_err))
            self.error_pos = error
        else:
            self.error_pos = np.concatenate([self.error_pos, error])

        self.error_it += 1

    def _compute_grf_th(self, contacts):

        c1 = contacts[0].wrench.force.z
        c2 = contacts[1].wrench.force.z
        c3 = contacts[2].wrench.force.z
        c4 = contacts[3].wrench.force.z

        self.c1_buff.append(c1)
        self.c2_buff.append(c2)
        self.c3_buff.append(c3)
        self.c4_buff.append(c4)

        c1_mean = sum(self.c1_buff) / len(self.c1_buff)
        c2_mean = sum(self.c2_buff) / len(self.c2_buff)
        c3_mean = sum(self.c3_buff) / len(self.c3_buff)
        c4_mean = sum(self.c4_buff) / len(self.c4_buff)

        # inp = [c1_mean if c1_mean > self.th_val else 0]
        # inp += [c2_mean if c2_mean > self.th_val else 0]
        # inp += [c3_mean if c3_mean  > self.th_val else 0]
        # inp += [c4_mean if c4_mean  > self.th_val else 0]

        inp = [self.th_val if c1_mean > self.th_val else 0]
        inp += [self.th_val if c2_mean > self.th_val else 0]
        inp += [self.th_val if c3_mean  > self.th_val else 0]
        inp += [self.th_val if c4_mean  > self.th_val else 0]


        self.c1_buff.pop(0)
        self.c2_buff.pop(0)
        self.c3_buff.pop(0)
        self.c4_buff.pop(0)

        try:
            self.pub_grf_th.publish(Float32MultiArray(data=inp))
        except:
            pass

        return inp

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

        if "grf_th" in self.hyq_inputs:
            inp += self._compute_grf_th(msg.contacts) # Processed Feet Z Forces

        if "joints" in self.hyq_inputs:
            inp += [msg.joints[1].position]          # LF Hip FE Joint
            inp += [msg.joints[2].position]          # LF Knee FE Joint
            inp += [msg.joints[4].position]          # RF Hip FE Joint
            inp += [msg.joints[5].position]          # RF Knee FE Joint
            inp += [msg.joints[7].position]          # LH Hip FE Joint
            inp += [msg.joints[8].position]          # LH Knee FE Joint
            inp += [msg.joints[10].position]         # RH Hip FE Joint
            inp += [msg.joints[11].position]         # RH Knee FE Joint

        if "imu" in self.hyq_inputs:
            inp += [msg.base[5].position]            # Base Z
            inp += [msg.base[0].position]            # Base Roll
            inp += [msg.base[1].position]            # Base Pitch

        self.process_state_flag.acquire()
        try:
            self.hyq_phi = msg.base[0].position
            self.hyq_theta = msg.base[1].position
            self.hyq_psi = msg.base[2].position
            self.hyq_x = msg.base[3].position
            self.hyq_y = msg.base[4].position
            self.hyq_z = msg.base[5].position
            self.hyq_lf_grf = msg.contacts[0].wrench.force.z
            self.hyq_rf_grf = msg.contacts[1].wrench.force.z
            self.hyq_lh_grf = msg.contacts[2].wrench.force.z
            self.hyq_rh_grf = msg.contacts[3].wrench.force.z
            if not self.hyq_fall and self.trot_started:
                if self.hyq_z < 0.35 or abs(self.hyq_phi) > 0.8 or abs(self.hyq_theta) > 0.8:
                    print " [Physics]   The robot has touched the ground because of Z={0:.2f}".format(self.hyq_z) + \
                          " or PHI={0:.2f}".format(self.hyq_phi) + " or THETA={0:.2f}".format(self.hyq_theta)
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

    def _reg_hyq_tgt_action(self, msg):

        # Robot Target Action (length = 8)
        out = [msg.position[1]]           # LF Hip FE Joint
        out += [msg.position[2]]          # LF Knee FE Joint
        out += [msg.position[4]]          # RF Hip FE Joint
        out += [msg.position[5]]          # RF Knee FE Joint
        out += [msg.position[7]]          # LH Hip FE Joint
        out += [msg.position[8]]          # LH Knee FE Joint
        out += [msg.position[10]]         # RH Hip FE Joint
        out += [msg.position[11]]         # RH Knee FE Joint


        out += [msg.velocity[1]]           # LF Hip FE Joint
        out += [msg.velocity[2]]          # LF Knee FE Joint
        out += [msg.velocity[4]]          # RF Hip FE Joint
        out += [msg.velocity[5]]          # RF Knee FE Joint
        out += [msg.velocity[7]]          # LH Hip FE Joint
        out += [msg.velocity[8]]          # LH Knee FE Joint
        out += [msg.velocity[10]]         # RH Hip FE Joint
        out += [msg.velocity[11]]         # RH Knee FE Joint

        self.process_action_flag.acquire()
        try:
            self.hyq_tgt_action = out
            self.hyq_full_tgt_action = copy.deepcopy(msg)
            self.hyq_action_it += 1
        finally:
            self.process_action_flag.release()

    def pub_mark(self):

        # Init publishers
        self.pub_lh_mark = ros.Publisher(self.hyq_lh_mark_pub_name, Marker,
                                         queue_size=1)
        self.pub_rh_mark = ros.Publisher(self.hyq_rh_mark_pub_name, Marker,
                                         queue_size=1)
        self.pub_lf_mark = ros.Publisher(self.hyq_lf_mark_pub_name, Marker,
                                         queue_size=1)
        self.pub_rf_mark = ros.Publisher(self.hyq_rf_mark_pub_name, Marker,
                                         queue_size=1)
        self.pub_base_mark = ros.Publisher(self.hyq_base_mark_pub_name, Marker,
                                         queue_size=1)
        self.pub_mark_rate = ros.Rate(25)
        self.pub_lh_points = []
        self.pub_rh_points = []
        self.pub_lf_points = []
        self.pub_rf_points = []
        self.pub_base_points = []

        # Markers values to be reused
        t = Marker.LINE_STRIP
        a = Marker.ADD
        d = ros.Duration(0.04)
        s = Vector3(0.01, 0.01, 0.01)
        h = Header(frame_id='world')
        c_lh = ColorRGBA(0.0, 1.0, 0.0, 1)
        c_rh = ColorRGBA(0.0, 0.4, 0.7, 1)
        c_lf = ColorRGBA(0.5, 0.0, 0.5, 1)
        c_rf = ColorRGBA(1.0, 0.0, 0.2, 1)
        c_base = ColorRGBA(0.0, 1.0, 0.0, 1)
        p = Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)) 

        while not self.t_stopped:

            try:
                # Get TFs
                (lh, r) = self.list_tf.lookupTransform('/world', '/lh_foot', ros.Time(0))
                (rh, r) = self.list_tf.lookupTransform('/world', '/rh_foot', ros.Time(0))
                (lf, r) = self.list_tf.lookupTransform('/world', '/lf_foot', ros.Time(0))
                (rf, r) = self.list_tf.lookupTransform('/world', '/rf_foot', ros.Time(0))
                (base, r) = self.list_tf.lookupTransform('/world', '/trunk', ros.Time(0))
                self.pub_lh_points.append(Point(lh[0], lh[1], lh[2]))
                self.pub_rh_points.append(Point(rh[0], rh[1], rh[2]))
                self.pub_lf_points.append(Point(lf[0], lf[1], lf[2]))
                self.pub_rf_points.append(Point(rf[0], rf[1], rf[2]))
                self.pub_base_points.append(Point(base[0], base[1], base[2]))

                # Publish markers
                m_lh = Marker(type=t, action=a, lifetime=d, pose=p, scale=s, header=h, color=c_lh,
                              points=self.pub_lh_points)
                m_rh = Marker(type=t, action=a, lifetime=d, pose=p, scale=s, header=h, color=c_rh,
                              points=self.pub_rh_points)
                m_lf = Marker(type=t, action=a, lifetime=d, pose=p, scale=s, header=h, color=c_lf,
                              points=self.pub_lf_points)
                m_rf = Marker(type=t, action=a, lifetime=d, pose=p, scale=s, header=h, color=c_rf,
                              points=self.pub_rf_points)
                m_base = Marker(type=t, action=a, lifetime=d, pose=p, scale=s, header=h, color=c_base,
                                points=self.pub_base_points)
                self.pub_lh_mark.publish(m_lh)
                self.pub_rh_mark.publish(m_rh)
                self.pub_lf_mark.publish(m_lf)
                self.pub_rf_mark.publish(m_rf)
                self.pub_base_mark.publish(m_base)

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass

            # Sleep
            self.pub_mark_rate.sleep()

    def draw_plane(self, add=False):

        if self.pub_poly is None:
            self.pub_poly = ros.Publisher(self.hyq_phase_poly_pub_name, Marker,
                                          queue_size=20)
            self.poly_points = []

        try:
            if add:
                x, y, z = self.get_hyq_x_y_z()

                t = Marker.TRIANGLE_LIST
                a = Marker.ADD
                d = ros.Duration(100)
                s = Vector3(1, 1, 1)
                h = Header(frame_id='world')
                c = ColorRGBA(0.0, 0.0, 1.0, 0.8)
                p = Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)) 
                self.poly_points += [Point(x, y - 0.7, z - 0.7), Point(x, y + 0.7, z + 0.1),
                                     Point(x, y - 0.7, z + 0.1), Point(x, y + 0.7, z + 0.1), 
                                     Point(x, y - 0.7, z - 0.7), Point(x, y + 0.7, z - 0.7)]

            m = Marker(type=t, action=a, lifetime=d, pose=p, scale=s, header=h,
                       color=c, points=self.poly_points)
            self.pub_poly.publish(m)

        except Exception:
            pass

    def printv(self, txt):

        if self.verbose >= 1:
            print(txt)


if __name__ == '__main__':

    # Create and start simulation
    p = HyQSim(verbose=2)  # init_impedance=[150, 12, 250, 8, 250, 6])
    p.start()
    p.register_node()

    if len(sys.argv) > 1:

        if sys.argv[1] == "sine":
            pose_init = np.array([-0.2, 0.76, -1.52,
                                  -0.2, 0.76, -1.52,
                                  -0.2, -0.76, 1.52,
                                  -0.2, -0.76, 1.52])
            while not ros.is_shutdown():
                if p.sim_started:
                    t = p.get_sim_time()
                    move_sin = 20 * np.pi / 180 * np.sin(2 * np.pi * 0.5 * t)
                    pose = pose_init + np.array([0, 0, move_sin, 0, 0, move_sin,
                                                 0, 0, -move_sin, 0, 0, -move_sin])
                    p.send_hyq_nn_pred(pose, 1)
                else:
                    time.sleep(0.01)

        if sys.argv[1] == "rcf":
            trot_flag = False
            for i in range(40):

                print("Time: " + str(i) + "s and Sim time: " +
                      str(p.get_sim_time()) + "s and state (len= " +
                      str(np.array(p.get_hyq_state()).shape) + "):\n" +
                      str(np.array(p.get_hyq_state())) + "\nAnd (x, y, z) = " +
                      str(p.get_hyq_x_y_z()))
                if trot_flag is False:
                    if p.sim_started:
                        trot_flag = p.start_rcf_trot()
                        if trot_flag:
                            print " ===== Trotting Started ====="
                if ros.is_shutdown():
                    p.stop()
                    exit(-1)
                time.sleep(1)

    else:
        for i in range(120):
            print("Time: " + str(i) + "s and Sim time: " +
                  str(p.get_sim_time()) + "s and state (len= " +
                  str(np.array(p.get_hyq_state()).shape) + "):\n" +
                  str(np.array(p.get_hyq_state())))
            if ros.is_shutdown():
                p.stop()
                exit(-1)
            time.sleep(1)

    # Stop simulation
    p.stop()
