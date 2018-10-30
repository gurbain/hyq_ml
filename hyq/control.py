import ast
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
import pickle as pkl
import rospy as ros
import sys
import time
import traceback

sine_config = {"params":
                    [{'f': 6, 'a': 0.4, 'phi': 0}, \
                    {'f': 6, 'a': 0.4, 'phi': 0}, \
                    {'f': 6, 'a': 0.4, 'phi': 3.14}, \
                    {'f': 6, 'a': 0.4, 'phi': 3.14}],
               "time_step": 0.001,
               "sim_time": 5}

cpg_config = {"params":
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
              "integ_time": 0.001,
              "time_step": 0.005,
              "sim_time": 10}


class Controller(object):

    def __init__(self, conf):

        # Retrieve the config
        self.config = conf
        self.params = self.config["params"]

        self.n_motors = len(self.params)
        self.openloop = True

        self.it = 0
        self.t = 0
        self.t_init = 0
        self.st_time_step = 0

        self.hist_cmd = []

    def __getstate__(self):
        """ This is called before pickling. """

        state = self.__dict__.copy()
        del state['hist_cmd']
        del state["log"]

        return state

    def __setstate__(self, state):
        """ This is called while unpickling. """

        self.__dict__.update(state)

    def get_params_len(self):

        length = 0
        for m in self.params:
            length += len(m)

        return length

    def get_norm_params(self):

        # Surcharge it depending on the controller
        return

    def set_norm_params(self, liste):

        # Surcharge it depending on the controller
        return

    def get_params(self):

        return self.params

    def set_params(self, params):

        self.params = params

    def step(self, t):
        """
        This function is called to update the controller state at each time_step
        """

        # This function must be surcharged

        self.t = t

        cmd = []
        for i in range(self.n_motors):
            cmd.append(0)

        return cmd

    def run(self, sim_time, physics):
        """
        This function is a blocking function that execute the controller for a given sim_time
        """

        st = 0

        try:

            self.t_init = time.time()

            # Wait for the physics to be started
            while 1:
                time.sleep(self.time_step)
                if physics.is_sim_started():
                    break

            while physics.sim_duration < sim_time:

                st = physics.sim_duration
                cmd = self.step(st)
                physics.set_sim_cmd(cmd)

                # Do something for real-time here

        except:
            ros.logerr("Simulation aborted by user. Physics time: " + str(physics.sim_duration) +
                           "s. Controller time: not set!")
            physics.kill_sim()
            traceback.print_exc()
            sys.exit()

        rt = time.time() - self.t_init
        ros.loginfo("Simulation of {0:.2f}s (ST)".format(st) +
                      " finished in {0:.2f}s (RT)".format(rt) +
                      " with acceleration of {0:.3f} x".format(st/rt))

    def load(self, filename):
        """
        Load itself from a pickle file
        """
        f = open(filename, 'rb')
        tmp_dict = pkl.load(f)
        f.close()

        self.__dict__.update(tmp_dict.__dict__)

    def save(self, filename):
        """
        Save class and variables with pickle
        """

        f = open(filename, 'wb')
        pkl.dump(self, f, 2)
        f.close()

    def plot(self, filename="history.png"):

        plt.plot(np.array(self.hist_cmd))
        plt.savefig(filename, format='png', dpi=300)
        plt.close()


class Sine(Controller):

    def __init__(self, conf):

        super(Sine, self).__init__(conf)

        self.norm_f = 2 * math.pi * self.params[0]["f"]
        self.norm_a = self.params[0]["a"]
        self.norm_phi = self.params[0]["phi"]

    def set_norm_params(self, liste):

        j = 0
        params = []
        for i in range(self.n_motors):
            f = liste[j] * self.norm_f
            a = liste[j+1] * self.norm_a
            phi = liste[j+2] * self.norm_phi
            params.append({"f": f, "a": a, "phi": phi})
            j += 3

        self.params = params

    def get_norm_params(self):

        liste = []
        for m in self.params:
            for i in m:
                liste.append(m)

        return liste

    def step(self, t):

        self.t = t
        cmd = []
        for i in range(self.n_motors):
            cmd.append(self.params[i]["a"] * math.sin(self.params[i]["f"] * self.t + self.params[i]["phi"]))

        self.hist_cmd.append(cmd)
        return cmd


class CPG(Controller):

    def __init__(self, conf):

        super(CPG, self).__init__(conf)

        self.r = [p["r"] for p in self.params]
        self.phi = [np.pi * float(self.params[i]["duty_factor"]) for i in range(self.n_motors)]
        self.o = [float(p["o"]) for p in self.params]

        self.kappa_r = [1] * self.n_motors
        self.kappa_phi = [1] * self.n_motors
        self.kappa_o = [1] * self.n_motors
        self.f_r = [0] * self.n_motors
        self.f_phi = [0] * self.n_motors
        self.f_o = [0] * self.n_motors

        self.dt = float(self.config["integ_time"])
        self.gamma = 0.1
        self.prev_t = -1

        if not all(["coupling" in p for p in self.params]):
            self.coupling = np.ones((self.n_motors, self.n_motors))
        else:
            self.coupling = np.array([p["coupling"] for p in self.params])

        psi_line = np.array([p["phase_offset"] for p in self.params])
        self.psi = np.zeros((self.n_motors, self.n_motors))
        for i in range(self.n_motors):
            self.psi[i, :] = psi_line - psi_line[i]

    def step_cpg(self):

        cmd = []

        for i in range(self.n_motors):

            # Fetch current motor values
            dt = self.dt
            gamma = self.gamma

            mu = float(self.params[i]["mu"])
            omega = float(self.params[i]["omega"])
            d = float(self.params[i]["duty_factor"])

            r = self.r[i]
            phi = self.phi[i]
            o = self.o[i]

            kappa_r = self.kappa_r[i]
            kappa_phi = self.kappa_phi[i]
            kappa_o = self.kappa_o[i]
            f_r = self.f_r[i]
            f_phi = self.f_phi[i]
            f_o = self.f_o[i]
            cpl = self.coupling[i]

            # Compute step evolution of r, phi and o
            d_r = gamma * (mu + kappa_r * f_r - r * r) * r
            d_phi = omega + kappa_phi * f_phi
            d_o = kappa_o * f_o

            # Add phase coupling
            for j in range(self.n_motors):
                d_phi += cpl[j] * np.sin(self.phi[j] - phi - self.psi[i][j])

            # Update r, phi and o
            self.r[i] += dt * d_r
            self.phi[i] += dt * d_phi
            self.o[i] += dt * d_o

            # Threshold phi to 2pi max
            phi_thr = 0
            phi_2pi = self.phi[i] % (2 * math.pi)
            if phi_2pi < ((2 * math.pi) * d):
                phi_thr = phi_2pi / (2 * d)
            else:
                phi_thr = (phi_2pi + (2 * math.pi) * (1 - 2 * d)) / (2 * (1 - d))

            # Save action
            action = self.r[i] * np.cos(phi_thr) + self.o[i]
            cmd.append(action)

        self.hist_cmd.append(cmd)
        return cmd

    def step(self, t):

        self.t = t
        n_steps = int(float(self.t)/self.dt - self.prev_t)
        cmd = []
        if n_steps == 0:
            n_steps = 1
            # ros.logerr("Controller time step (" + str(float(self.t)/self.dt - self.prev_t) +
            #                "ms) is too low for numerical integration (dt = " + str(self.dt*1000) + " ms). " +
            #                "Truncating control signal to avoid stopping software!")

        for _ in range(n_steps):
            cmd = self.step_cpg()

        self.prev_t = float(self.t/self.dt)

        return cmd



if __name__ == '__main__':

    # Test CPG evolution
    t = 0
    dt = cpg_config["time_step"]
    st = cpg_config["sim_time"]
    n = int(st / dt)
    sc = CPG(cpg_config)

    t_init = time.time()
    for i in range(n):
        sc.step(t)
        t += dt
    t_tot = time.time() - t_init
    print(str(n) + " iterations computed in " + str(t_tot) + " s")

    sc.plot("cpg.png")
