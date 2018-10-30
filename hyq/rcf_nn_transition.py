
import rospy as ros
from std_srvs.srv import Trigger

import thread
import time

# Create node and service
ros.init_node('sim_transition_node', anonymous=True)
service = ros.ServiceProxy("/sim/rcf_nn_transition", Trigger)

# Listen to keyboard in a loop
while not ros.is_shutdown():
    c = raw_input("To force the RCF/NN transition, press ENTER: ")
    ack_service = service()
    print ack_service
