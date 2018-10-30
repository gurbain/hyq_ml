import os
import rospy as ros
import socket
import sys
import threading
import time

import utils
import physics


class SimServer(object):

    def __init__(self, ip='0.0.0.0', port=4800):

        self.ip = str(ip)
        self.port = int(port)

        self.server = None

    def start(self):

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.ip, self.port))
        self.server.listen(1)  # max backlog of connections

        print 'Listening on {}:{}'.format(self.ip, self.port)

    def run(self):

        while True:
            if self.server is not None:
                client_sock, address = self.server.accept()
                print 'Accepted connection from {}:{}'.format(address[0], address[1])
                client_handler = threading.Thread(target=self.handle_conn, args=(client_sock,))
                client_handler.start()


    def handle_conn(self, client_socket):

        rqt = client_socket.recv(1024)
        print 'Received {}'.format(request)
        client_socket.send('ACK!')
        client_socket.close()

    def simulate(n, options):

        # Create and start simulation
        if not ros.is_shutdown():
            p = physics.HyQSim(verbose=1, init_impedance=[kp, kd, kp, kd, kp, kd])
            p.start()

        # Run Simulation for 40 seconds
        t = 0  # sim time
        i = 0  # real timeout
        x = []  # robot init x position
        y = []  # robot init y position
        t_trot = 0 # start of the trotting gait
        trot_flag = False
        while not ros.is_shutdown() and t < 50 and i < 2000:

            # Get simulation time
            t = p.get_sim_time()

            # Start the trot
            if trot_flag is False:
                if p.sim_started:
                    trot_flag = p.start_rcf_trot()
                    if trot_flag:
                        print " ===== Trotting Started =====\n"
                        t_trot = t

            # Retrieve robot x and y position
            curr_x, curr_y = p.get_hyq_xy()
            x.append(curr_x)
            y.append(curr_y)

            # Count for timeout
            i += 1
            time.sleep(0.1)

        # Get result and stop simulation
        if 'p' in locals():
            p.stop()

            # Save and quit
            to_save = {"t_sim": t, "x": x, "y": y, "t_real": i,
                       "t_trot": t_trot, "index": n, "Kp": kp, "Kd": kd}
            utils.save_on_top(to_save, FOLDER + "results.pkl")


if __name__ == '__main__':

    if len(sys.argv) > 1:

        s = SimServer(port=sys.argv[1])
        s.start()
        s.run()

