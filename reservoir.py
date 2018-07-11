import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import time


## MATPLOTLIB STYLE
plt.style.use('fivethirtyeight')
plt.rc('lines', linewidth= 1)
plt.rc('text', usetex=False)
plt.rc('axes', facecolor='white')
plt.rc('savefig', facecolor='white')
plt.rc('figure', autolayout=True)


class Neuron(object):

    def tran_fct(self, x):

        return np.tanh(x)

        # x_temp = x.copy()
        # for idx, itm in enumerate(x_temp):
        #     if itm <= 0 :
        #         x_temp[idx] = 0
        #     else :
        #         x_temp[idx] = 0.786*itm/(0.110+itm)-0.014

        # return x_temp


class ReservoirNet(Neuron):
    """
    This class implements constant and methods of a artificial neural network
    using Reservoir Computing architecture, with the goal of comparing with a network of spiking populations.
    """

    def __init__(self, n_in=0, n_fb=0, n_res=200, spec_rad=0.9, sparse=False,
                 leakage=0.1, scale_bias=0., scale_fb=5.0, verbose=3,
                 neg_w=True, seed=None, keep_hist=True):

        Neuron.__init__(self)

        self.scale_bias = scale_bias
        self.scale_fb = scale_fb
        self.spec_rad = spec_rad
        self.sparse_con = sparse
        self.leakage = leakage
        self.seed = seed
        self.neg_w = neg_w
        self.verbose = verbose
        self.keep_hist = keep_hist

        self.n_in = n_in
        self.n_fb = n_fb
        self.n_res = n_res

    def start(self):

        self.printv("\n\n ===== Initializing Reservoir =====\n")

        ti = time.time()

        # Get random generator
        if self.seed == None:
            self.rng = np.random.RandomState(np.random.randint(0,99999))
        else:
            self.rng = np.random.RandomState(self.seed)
        t1 = time.time()
        print t1-ti

        # Initialize input weights
        self.w_in = self.rng.randn(self.n_res, self.n_in)
        if not(self.neg_w):
            self.w_in = abs(self.w_in)

        # Initialise bias
        self.w_bias = self.rng.randn(self.n_res, 1) * self.scale_bias
        t2 = time.time()
        print t2-t1

        # Initialize reservoir random weights given spectral radius
        self.w_res = self.get_rand_mat()
        t3 = time.time()
        print t3-t2

        # Initialize reservoir feedback (not used)
        self.w_fb = self.rng.randn(self.n_res, self.n_fb) * self.scale_fb
        if not(self.neg_w):
            self.w_fb = abs(self.w_fb)
        t4 = time.time()
        print t4-t3

        # Create connection probability
        if self.sparse_con:
            self.p_connect_res = self.rng.randint(2,
                                 size=(self.n_res, self.n_res))
        self.p_connect_fb = 0.1
        self.p_connect_in = 1.0
        t5 = time.time()
        print t5-t4

        # Create arrays of zeros for the initial states
        self.x = np.zeros((self.n_res, 1))
        self.u = np.zeros(self.n_in)

        # Set iteration number to 0
        self.it = 0
        self.t_compute_in = 0
        self.t_compute_res = 0
        self.t_compute_tf = 0
        self.t_compute_hist = 0

    def get_coord(self, ind, x_d, y_d):

        if not (isinstance(ind, (int, long)) & isinstance(x_d, (int, long)) & isinstance(y_d, (int, long))):
            raise Exception('population index, x dimension and y dimension must be integer types')
        z_d = x_d * y_d

        z = ind / z_d
        y = (ind - z * z_d) / x_d
        x = ind - z * z_d - y * x_d

        return x, y, z

    def get_prob(self, id_0, id_1, x_d, y_d, c=0.3, lamb=1.0):

        if id_0 == id_1:
            prob = 0.
        else:
            x_0, y_0, z_0 = self.get_coord(id_0, x_d, y_d)
            x_1, y_1, z_1 = self.get_coord(id_1, x_d, y_d)
            d = np.sqrt((x_0 - x_1) ** 2 +
                        (y_0 - y_1) ** 2 + (z_0 - z_1) ** 2)  # eucl distance

            prob = c * np.power(np.e, -np.square(d / lamb))

        return prob

    def create_conn_mat(self):

        p_connect = np.empty((self.n_res, self.n_res))

        for fr in range(self.n_res):
            for to in range(self.n_res):
                p_connect[fr, to] = self.get_prob(to, fr, x_d=3, y_d=3)

        return p_connect

    def get_rand_mat(self):

        mat = self.rng.randn(self.n_res, self.n_res)
        if not(self.neg_w):
            mat = abs(mat)

        w, v = np.linalg.eig(mat)
        mat = np.divide(mat, (np.amax(np.absolute(w)) / self.spec_rad))

        return mat

    def step(self, u=None, y=None):

        # Print status and timing
        if self.verbose > 2 and self.it % 2000 == 0 and self.it !=0:
            print "Iteration " + str(self.it) + \
                  ": In: {:.2f}".format(self.t_compute_in) + \
                  "s; Res: {:.2f}".format(self.t_compute_res) + \
                  "s; TF: {:.2f}".format(self.t_compute_tf) + \
                  "s; Hist: {:.2f}".format(self.t_compute_hist)
            self.t_compute_in = 0
            self.t_compute_res = 0
            self.t_compute_tf = 0
            self.t_compute_hist = 0

        # Compute input value
        self.u = u
        ti = time.time()
        inp = np.mat(np.dot(self.w_in, self.u) * self.p_connect_in).T
        t1 = time.time()

        # Apply connection probability if relevant
        if self.sparse_con:
            self.w_res = self.w_res * self.p_connect_res

        # Compute reservoir update
        x_temp = np.dot(self.w_res, self.x) + \
                 self.w_bias  + inp
        t2 = time.time()

        # Apply transfer function with leakage
        self.x = (1 - self.leakage) * self.x + \
                 self.leakage * self.tran_fct(x_temp)
        t3 = time.time()

        # Save data in history
        if self.keep_hist:
            if self.it == 0:
                self.x_hist = self.x.T.tolist()
                self.u_hist = self.u.T.tolist()
            else:
                self.x_hist.append(self.x.T.tolist()[0])
                self.u_hist.append(self.u.T.tolist()[0])
        t4 = time.time()

        self.it += 1
        self.t_compute_in += t1 - ti
        self.t_compute_res += t2 - t1
        self.t_compute_tf += t3 - t2
        self.t_compute_hist += t4 - t3

        return self.x

    def run(self, u=None):

        # Random initialization of reservoir weights
        self.start()
        num_it = u.shape[0]

        # Process the input step by step
        self.printv("\n\n ===== Updating Reservoir =====\n")
        for i in range(num_it):
            self.step(u[i, :])

        self.x_hist = np.matrix(self.x_hist)

        return self.x_hist

    def printv(self, txt):

        if self.verbose > 1:
            print(txt)

    def plot_sum_states(self, u=None):

        self.x_hist = np.matrix(self.x_hist)
        s = np.sum(self.x_hist, axis=1) / self.x_hist.shape[1]
        plt.plot(s, label="sum of reservoir states")

        if u is not None:
            u_sum = np.sum(u, axis=1) / u.shape[1]
            plt.plot(u_sum, label="sum of inputs")

        plt.legend()
        plt.show()

    def plot_states(self, u=None):

        max_states_plotted = 10

        self.x_hist = np.matrix(self.x_hist)
        n_res = min(self.x_hist.shape[1], max_states_plotted)
        for i in range(n_res):
            plt.plot(self.x_hist[:, i] + i, color="b")

        n_u = min(u.shape[1], max_states_plotted)
        if u is not None:
            for i in range(n_u):
                plt.plot(u[:, i] + i, color="r")
        plt.show()

if __name__ == "__main__":

    if len(sys.argv) > 1:

        if sys.argv[1] == "test_impulse":
            # Create an impulse array
            inputs = np.zeros((4000, 25))
            inputs[1100:1110, :] = 1.0
            inputs[2300:2350, :] = 1.0
            n_in = inputs.shape[1]

            # Create and run reservoir
            r = ReservoirNet(n_in=n_in)
            r.run(inputs)
            r.plot_sum_states()
            r.plot_states()

        if sys.argv[1] == "test_data":

            with open(sys.argv[2] , "rb") as f:
                [x, y] = pickle.load(f)

            # Create and run reservoir
            r = ReservoirNet(n_in=x.shape[1])
            r.run(x)
            r.plot_states(x)
            r.plot_sum_states(x)


