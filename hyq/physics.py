
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
import threading
import traceback

from gazebo_msgs.srv import ApplyBodyWrench
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

    def __init__(self, view=False, rviz=False, rt=True, remote=False,
                 init_impedance=None, verbose=2, publish_error=False):

        threading.Thread.__init__(self)

        # Process information
        self.init_impedance = init_impedance
        self.remote = remote
        self.sim_ps = None
        self.sim_ps_name = "roslaunch"
        self.sim_package = "dls_supervisor"
        self.sim_node = "operator.launch"
        self.sim_to_kill = ["ros", "gzserver", "gzclient"]
        self.sim_not_kill = ["PlotJuggler", "rosmaster", "rosout", "roscore"]
        self.sim_params = "gazebo:=True osc:=False"
        if view is False:
            self.sim_params += " gui:=False"
        if rviz is False:
            self.sim_params += " rviz:=False"
        self.rcf_config_file = "/home/gurbain/hyq_ml/config/hyq_sim_options.ini"

        # ROS Topics
        self.clock_sub_name = 'clock'
        self.hyq_state_sub_name = "/hyq/robot_states"
        self.hyq_tgt_action_sub_name = "/hyq/des_rcf_joint_states"
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
        self.sub_tgt_action = None
        self.pub_nn = None
        self.pub_nn_w = None
        self.pub_js_err = None
        self.pub_haa_err = None
        self.pub_hfe_err = None
        self.pub_kfe_err = None
        self.pub_haad_err = None
        self.pub_hfed_err = None
        self.pub_kfed_err = None
        self.pub_tot_err = None
        self.srv_noise = None

        # Simulation state
        self.sim_time = 0
        self.hyq_state = None
        self.hyq_tgt_action = None
        self.hyq_x = 0
        self.hyq_y = 0
        self.hyq_state_it = 0
        self.hyq_action_it = 0
        self.process_state_flag = threading.Lock()
        self.process_action_flag = threading.Lock()

        # Class behaviour
        self.rt = rt
        self.t_init = time.time()
        self.controller_started = False
        self.sim_started = False
        self.publish_error = publish_error
        self.error_it = 0
        self.error_pos = np.array([])
        self.error_vel = np.array([])
        self.noise_it = 0
        self.next_noise_it = 0
        self.verbose = verbose
        self.daemon = True
        self.stopped = False

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

            # Create ROS Publishers
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
            # Start the physics
            self.start_sim()

            # Wait and start the controller
            self.start_controller()

            self.sim_started = True

        except Exception, e:
            if self.verbose > 0:
                print "Exception encountered during simulation!"  # + str(e)
                # traceback.print_exc()
            if self.sim_ps is not None:
                if self.sim_ps.isalive():
                    self.stop_sim()

        while not self.stopped:
            time.sleep(0.1)

    def stop(self):

        self.stop_sim()
        self.stopped = True

    def start_sim(self):

        # If the physics should not on this computer, just do not do anything
        if self.remote:
            self.printv(" ===== Physics should run on remote computer =====\n")
            return

        try:
            # Init the config file
            if self.init_impedance is not None:
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
                print "Exception encountered during simulation!"  + str(e)
                traceback.print_exc()
            if self.sim_ps is not None:
                if self.sim_ps.isalive():
                    self.stop_sim()
    
    def stop_sim(self, clean_stop=False):

        if self.remote:
            return

        self.printv(" ===== Stopping Physics =====\n")

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
                for proc in psutil.process_iter():
                    name = " ".join(proc.cmdline())
                    for y in self.sim_to_kill:
                        if y in name:
                            kill_me = True
                            for n in self.sim_not_kill:
                                if n in name:
                                    kill_me = False
                            if kill_me:
                                try:
                                    proc.kill()
                                except Exception, e:
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
                    if self.verbose > 1:
                        print(str(self.sim_ps.before))
                pass

    def start_controller(self):

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
            return True

        else:
            return False

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

        f = fileinput.FileInput(self.rcf_config_file, inplace=True, backup='.bak')

        for line in f:
            if line.startswith("KpHAA = "):
                print "KpHAA = {0:.3f}".format(float(self.init_impedance[0]))
            elif line.startswith("KdHAA = "):
                print "KdHAA = {0:.3f}".format(float(self.init_impedance[1]))
            elif line.startswith("KpHFE = "):
                print "KpHFE = {0:.3f}".format(float(self.init_impedance[2]))
            elif line.startswith("KdHFE = "):
                print "KdHFE = {0:.3f}".format(float(self.init_impedance[3]))
            elif line.startswith("KpKFE = "):
                print "KpKFE = {0:.3f}".format(float(self.init_impedance[4]))
            elif line.startswith("KdKFE = "):
                print "KdKFE = {0:.3f}".format(float(self.init_impedance[5]))
            else:
                print line.rstrip()

        f.close()

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
            return self.sim_time
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

    def get_hyq_xy(self):

        return copy.deepcopy(self.hyq_x), copy.deepcopy(self.hyq_y)

    def send_hyq_nn_pred(self, prediction, weight, error=None):

        if type(prediction) is np.ndarray:
            prediction = prediction.tolist()

        if len(prediction) != 12:
                ros.logerr("This method is designed to receive 12 joint" +
                           " position and velocity in a specific format!")
                return

        # Create and fill a JointState object
        joints = JointState()
        joints.header = Header()
        if self.rt:
            joints.header.stamp = ros.get_rostime()
        else:
            joints.header.stamp = ros.Time(self.get_sim_time())
        joints.position = prediction[0:12]
        joints.velocity = [0.0] * 12
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

        if self.noise_it == self.next_noise_it:
            noise = Wrench()
            noise.force.x = random.uniform(-50, 50)
            noise.force.y = random.uniform(-50, 50)
            noise.force.z = random.uniform(-50, 50)

            try:
                self.srv_noise("hyq::base_link", "", None, noise, ros.Time.now(), ros.Duration.from_sec(1.0))
            except:
                pass

            self.next_noise_it += random.randint(2000, 15000)

        self.noise_it += 1

    def publish_errs(self, error):

        # Always publish the joint error
        joints_err = JointState()
        joints_err.header = Header()
        joints_err.header.stamp = ros.Time(self.get_sim_time())
        joints_err.position = error
        joints_err.velocity = [0.0] * 12
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

    def _reg_sim_time(self, time):

        self.sim_time = time.clock.secs + time.clock.nsecs/1000000000.0

    def _reg_hyq_state(self, msg):

        # Robot State (length = 24)
        inp = [0.5]                              # Bias
        inp += [msg.base[5].position]            # Base Z
        inp += [msg.base[0].position]            # Base Roll
        inp += [msg.base[1].position]            # Base Pitch

        inp += [msg.contacts[0].wrench.force.x]  # LF Foot X Force
        inp += [msg.contacts[0].wrench.force.y]  # LF Foot Y Force
        inp += [msg.contacts[0].wrench.force.z]  # LF Foot Z Force
        inp += [msg.contacts[1].wrench.force.x]  # RF Foot X Force
        inp += [msg.contacts[1].wrench.force.y]  # RF Foot Y Force
        inp += [msg.contacts[1].wrench.force.z]  # RF Foot Z Force
        inp += [msg.contacts[2].wrench.force.x]  # LH Foot X Force
        inp += [msg.contacts[2].wrench.force.y]  # LH Foot Y Force
        inp += [msg.contacts[2].wrench.force.z]  # LH Foot Z Force
        inp += [msg.contacts[3].wrench.force.x]  # RH Foot X Force
        inp += [msg.contacts[3].wrench.force.y]  # RH Foot Y Force
        inp += [msg.contacts[3].wrench.force.z]  # RH Foot Z Force

        inp += [msg.joints[1].position]          # LF Hip FE Joint
        inp += [msg.joints[2].position]          # LF Knee FE Joint
        inp += [msg.joints[4].position]          # RF Hip FE Joint
        inp += [msg.joints[5].position]          # RF Knee FE Joint
        inp += [msg.joints[7].position]          # LH Hip FE Joint
        inp += [msg.joints[8].position]          # LH Knee FE Joint
        inp += [msg.joints[10].position]         # RH Hip FE Joint
        inp += [msg.joints[11].position]         # RH Knee FE Joint

        self.process_state_flag.acquire()
        try:
            self.hyq_x = msg.base[3].position
            self.hyq_y = msg.base[4].position
            self.hyq_state = inp
            self.hyq_state_it += 1
        finally:
            self.process_state_flag.release()

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

        self.process_action_flag.acquire()
        try:
            self.hyq_tgt_action = out
            self.hyq_action_it += 1
        finally:
            self.process_action_flag.release()

    def printv(self, txt):

        if self.verbose >= 1:
            print(txt)


if __name__ == '__main__':

    # Create and start simulation
    p = HyQSim(init_impedance=[150, 12, 250, 8, 250, 6])
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
            for i in range(120):

                print("Time: " + str(i) + "s and Sim time: " +
                      str(p.get_sim_time()) + "s and state (len= " +
                      str(np.array(p.get_hyq_state()).shape) + "):\n" +
                      str(np.array(p.get_hyq_state())) + "\nAnd (x, y) = " +
                      str(p.get_hyq_xy()))
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