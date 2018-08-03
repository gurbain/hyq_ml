
import copy
import numpy as np
import os
import pexpect
import psutil
import rospy as ros
import rosgraph
import sys
import time
import threading
import traceback

from geometry_msgs.msg import Vector3, Wrench
from rosgraph_msgs.msg import Clock
from dls_msgs.msg import StringDoubleArray
from dwl_msgs.msg import WholeBodyState, WholeBodyTrajectory, JointState, ContactState, BaseState
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import utils
sys.stderr = stderr


class HyQSim(threading.Thread):

    def __init__(self, view=False, verbose=2):

        threading.Thread.__init__(self)

        # Process information
        self.sim_ps = None
        self.sim_ps_name = "roslaunch"
        self.sim_package = "dls_supervisor"
        self.sim_node = "hyq.launch"
        self.sim_to_kill = "ros"
        self.sim_not_kill = "plotjuggler"
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
        self.hyq_action_sub_name = "/hyq/des_joint_states"
        self.hyq_debug_sub_name = "/hyq/debug"
        self.hyq_plan_pub_name = "/hyq/plan"
        self.sub_clock = None
        self.sub_state = None
        self.sub_action = None
        self.sub_debug = None
        self.pub_plan = None

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

            # Write it in the actual value
            traj.actual = curr

            # Transform the actual trajectory with the changed values
            traj.trajectory = self.hyq_traj

            # Publish the message
            self.pub_plan.publish(traj)
            self.hyq_traj_it += 1

        # Flush the trajectory EVEN if not sent
        self.hyq_traj = []

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